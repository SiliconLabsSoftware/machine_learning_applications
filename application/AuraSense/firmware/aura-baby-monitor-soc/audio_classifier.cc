/**************************************************************************//**
 * @file
 * @brief Audio classifier — 2-class (laugh / sad) with 4-output TFLite model
 *
 * Model output layout (1, 4) int8 — ALL values 0.0-1.0 after dequant:
 *   [0] P(CRY)
 *   [1] P(NOT_CRY)
 *   [2] P(SAD)
 *   [3] P(LAUGH)
 *
 * Decision:
 *   1. If P(CRY) < CRY_CONF_THRESHOLD → background
 *   2. Else if P(SAD) >= P(LAUGH)     → sad
 *   3. Else                           → laugh
 *
 * LED mapping:
 *   Laugh → LED0 + LED1 (yellow)
 *   Sad   → LED2        (red)
 *   BG    → all off
 *****************************************************************************/
#include "os.h"
#include "sl_power_manager.h"
#include "sl_status.h"
#include "sl_led.h"
#include "sl_simple_led_instances.h"
#include "audio_classifier.h"
#include "app.h"
#include "config/audio_classifier_config.h"
#include "sl_board_control.h"
#include "sl_tflite_micro_model.h"
#include "sl_tflite_micro_init.h"
#include "sl_ml_audio_feature_generation.h"
#include "sl_sleeptimer.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include "config/sl_ml_audio_feature_generation_config.h"

#if SL_SIMPLE_LED_COUNT < 2
#error "Sample application requires at least two leds"
#endif

static OS_TCB tcb;
static CPU_STK stack[TASK_STACK_SIZE];
static void audio_classifier_task(void *arg);

// ── Thresholds ─────────────────────────────────────────────────────────────
// Stage-1 cry gate hysteresis based on model output[0] = P(CRY).
// Enter cry-state at a higher threshold, exit at a lower threshold to
// suppress flicker when p_cry jitters frame-to-frame.
#define CRY_CONF_ENTER_THRESHOLD 0.25f
#define CRY_CONF_EXIT_THRESHOLD  0.10f

// Feature variance gate — silences very flat (silent) frames.
#define FEATURE_VARIANCE_THRESHOLD  2000

// BLE notification interval
#define BLE_NOTIFY_INTERVAL_MS      1000
#define CONTROL_IDLE_INTERVAL_MS    200
#define DEEP_SLEEP_IDLE_INTERVAL_MS 1000

// Rolling majority window — mirrors Python ContinuousOutputEngine
// 6 steps x 0.5s = 3s window, same as WINDOW_SEC in Colab
#define MAJORITY_WINDOW_SIZE        6
#define MAJORITY_MIN_CRY_COUNT      1

// Additional detector consistency check:
// classify emotion only when cry head is meaningfully above not-cry head.
#define CRY_OVER_NOT_CRY_MARGIN     0.05f
// Close the cry gate only after this many consecutive non-cry frames.
#define NON_CRY_CLOSE_FRAMES        4

// ── Class IDs ──────────────────────────────────────────────────────────────
#define NUM_CLASSES     2
#define NO_CLASS_ID     255u
#define CLASS_LAUGH     0
#define CLASS_SAD       1
#define BLE_CLASS_BACKGROUND 0u
#define BLE_CLASS_LAUGH      1u
#define BLE_CLASS_SAD        2u

// ── Output tensor index mapping ────────────────────────────────────────────
#define OUT_IDX_CRY           0
#define OUT_IDX_NOT_CRY       1
#define OUT_IDX_SAD           2
#define OUT_IDX_LAUGH         3
#define OUT_TOTAL             4

static int gated_printf(const char *fmt, ...)
{
  if (!device_enabled || deep_sleep_enabled) {
    return 0;
  }
  va_list args;
  va_start(args, fmt);
  int written = vprintf(fmt, args);
  va_end(args);
  return written;
}

#define printf(...) gated_printf(__VA_ARGS__)

// ── State ──────────────────────────────────────────────────────────────────
static uint8_t  majority_ring[MAJORITY_WINDOW_SIZE];  // stores CLASS_LAUGH / CLASS_SAD / NO_CLASS_ID
static int      majority_head  = 0;
static int      majority_count = 0;

static uint32_t last_ble_notify_tick = 0;
static bool     input_shape_warned   = false;
static bool     cry_gate_open        = false;
static uint8_t  non_cry_streak       = 0;
static bool     em1_requirement_held = false;

static const char* CLASS_NAMES[NUM_CLASSES] = { "LAUGH", "SAD" };

// ── Feature extraction constants ───────────────────────────────────────────
#define MAX_MONO_FEATURE_ELEMENTS   4096
#define DYNAMIC_RANGE_Q             300
#define FLOAT_QUANT_FALLBACK_SCALE  (1.0f / 128.0f)

// ── Helpers ────────────────────────────────────────────────────────────────
static inline int tensor_element_size_bytes(const TfLiteTensor* tensor)
{
  if (tensor == NULL) return 0;
  if (tensor->type == kTfLiteInt8)    return (int)sizeof(int8_t);
  if (tensor->type == kTfLiteFloat32) return (int)sizeof(float);
  return 0;
}

static inline int tensor_element_count(const TfLiteTensor* tensor)
{
  const int elem_size = tensor_element_size_bytes(tensor);
  if (elem_size <= 0 || tensor == NULL || tensor->bytes <= 0) return 0;
  return (int)(tensor->bytes / elem_size);
}

static inline int8_t clamp_q7(int value)
{
  if (value >  127) return  127;
  if (value < -128) return -128;
  return (int8_t)value;
}

static uint32_t ticks_to_ms(uint32_t ticks)
{
  const uint32_t freq_hz = sl_sleeptimer_get_timer_frequency();
  if (freq_hz == 0u) return 0u;
  return (uint32_t)(((uint64_t)ticks * 1000u) / (uint64_t)freq_hz);
}

// ── Dequantize one output element to float 0.0-1.0 ─────────────────────────
// Uses the tensor's own scale + zero_point from PTQ calibration.
static float dequant_output(const TfLiteTensor* output, int index)
{
  if (output == NULL || index < 0) return 0.0f;

  float val = 0.0f;
  if (output->type == kTfLiteInt8) {
    const float scale = (output->params.scale > 0.0f)
                        ? output->params.scale : (1.0f / 127.0f);
    const int   zp    = output->params.zero_point;
    val = ((float)output->data.int8[index] - (float)zp) * scale;
  } else if (output->type == kTfLiteFloat32) {
    val = output->data.f[index];
  }

  // Clamp to [0, 1] — all our outputs should already be in this range
  if (val < 0.0f) val = 0.0f;
  if (val > 1.0f) val = 1.0f;
  return val;
}

// ── Input quantization helper ───────────────────────────────────────────────
static inline int8_t quantize_input_float(float v, const TfLiteTensor* input)
{
  float scale = FLOAT_QUANT_FALLBACK_SCALE;
  int   zp    = 0;
  if (input != NULL) {
    if (input->params.scale > 0.0f) scale = input->params.scale;
    zp = input->params.zero_point;
  }
  int q = (int)lrintf((v / scale) + (float)zp);
  if (q < -128) q = -128;
  else if (q > 127) q = 127;
  return (int8_t)q;
}

// ── 3-channel feature fill (mel + flatness + RMS) ──────────────────────────
static sl_status_t fill_input_tensor_3ch_from_features(TfLiteTensor* input,
                                                       int feature_elements)
{
  if (input == NULL || feature_elements <= 0 || feature_elements > MAX_MONO_FEATURE_ELEMENTS) {
    return SL_STATUS_INVALID_PARAMETER;
  }

  const int mel_bins = SL_ML_FRONTEND_FILTERBANK_N_CHANNELS;
  if (mel_bins <= 0 || (feature_elements % mel_bins) != 0) {
    return SL_STATUS_INVALID_PARAMETER;
  }
  const int frames = feature_elements / mel_bins;

  static uint16_t raw_buffer[MAX_MONO_FEATURE_ELEMENTS];
  static int8_t   ch0_buffer[MAX_MONO_FEATURE_ELEMENTS];
  static float    flatness_per_frame[128];
  static float    rms_per_frame[128];

  if (frames > (int)(sizeof(flatness_per_frame) / sizeof(flatness_per_frame[0]))) {
    return SL_STATUS_INVALID_PARAMETER;
  }

  sl_status_t status = sl_ml_audio_feature_generation_get_features_raw(
    raw_buffer, (size_t)feature_elements);
  if (status != SL_STATUS_OK) return status;

  // ── Channel 0: dynamic-range mel quantization ─────────────────────────
  int32_t maxval = 0;
  for (int i = 0; i < feature_elements; i++) {
    int32_t v = (int32_t)raw_buffer[i];
    if (v > maxval) maxval = v;
  }
  int32_t minval = maxval - DYNAMIC_RANGE_Q;
  if (minval < 0) minval = 0;
  int32_t val_range = maxval - minval;
  if (val_range < 1) val_range = 1;

  for (int i = 0; i < feature_elements; i++) {
    int32_t v = (int32_t)raw_buffer[i] - minval;
    v = (v * 255) / val_range;
    v -= 128;
    if (v < -128) v = -128;
    else if (v > 127) v = 127;
    ch0_buffer[i] = (int8_t)v;
  }

  // ── Channels 1 & 2: spectral flatness and RMS per frame ──────────────
  // MUST match Python extract_features():
  //   flatness: log1p(flat * 1000) → normalize by max → [0, 1]
  //   rms:      log1p(rms * 100)   → normalize by max → [0, 1]
  // Previous code used (x * 2 - 1) remapping to [-1,1] which DESTROYED
  // the bottom half of the int8 range (all clamped to -128).
  const float inv_scale = 1.0f / (float)(1 << SL_ML_FRONTEND_LOG_SCALE_SHIFT);
  float flat_max = 1e-9f;
  float rms_max  = 1e-9f;

  for (int f = 0; f < frames; f++) {
    float sum_lx = 0.0f;
    float max_lx = -1e9f;

    for (int m = 0; m < mel_bins; m++) {
      const int   idx = f * mel_bins + m;
      const float lx  = (float)raw_buffer[idx] * inv_scale;
      sum_lx += lx;
      if (lx > max_lx) max_lx = lx;
    }

    // log-sum-exp for log(arithmetic_mean(power))
    float sum_exp = 0.0f;
    for (int m = 0; m < mel_bins; m++) {
      const int   idx = f * mel_bins + m;
      const float lx  = (float)raw_buffer[idx] * inv_scale;
      sum_exp += expf(lx - max_lx);
    }
    const float log_arith = max_lx + logf(sum_exp / (float)mel_bins);
    const float mean_log  = sum_lx / (float)mel_bins;

    // Spectral flatness: geometric_mean / arithmetic_mean ∈ [0, 1]
    float flat = expf(mean_log - log_arith);
    if (flat < 0.0f) flat = 0.0f;
    else if (flat > 1.0f) flat = 1.0f;
    // Match Python: np.log1p(flatness * 1000.0)
    flat = logf(1.0f + flat * 1000.0f);
    flatness_per_frame[f] = flat;
    if (flat > flat_max) flat_max = flat;

    // RMS energy from log-power
    float rms = expf(0.5f * log_arith);
    // Match Python: np.log1p(rms * 100.0)
    rms = logf(1.0f + rms * 100.0f);
    rms_per_frame[f] = rms;
    if (rms > rms_max) rms_max = rms;
  }

  // Normalize flatness to [0, 1] — match Python: flatness / max(flatness)
  for (int f = 0; f < frames; f++) {
    float fl = flatness_per_frame[f] / flat_max;
    if (fl < 0.0f) fl = 0.0f;
    else if (fl > 1.0f) fl = 1.0f;
    flatness_per_frame[f] = fl;
  }

  // Normalize RMS to [0, 1] — match Python: rms / max(rms)
  for (int f = 0; f < frames; f++) {
    float r = rms_per_frame[f] / rms_max;
    if (r < 0.0f) r = 0.0f;
    else if (r > 1.0f) r = 1.0f;
    rms_per_frame[f] = r;
  }

  // ── Pack NHWC int8: (frames, mel_bins, 3) ────────────────────────────
  if (input->type != kTfLiteInt8) return SL_STATUS_INVALID_PARAMETER;

  for (int f = 0; f < frames; f++) {
    const int   base   = f * mel_bins;
    const int8_t flat_q = quantize_input_float(flatness_per_frame[f], input);
    const int8_t rms_q  = quantize_input_float(rms_per_frame[f],      input);
    for (int m = 0; m < mel_bins; m++) {
      const int idx = base + m;
      // Ch0: ch0_buffer is already [-128,127] matching the model's int8 range
      // (0.0 in Python → -128, 1.0 in Python → 127)
      // Direct assignment is correct — previous quantize_input_float(v/127)
      // mapped [-128,127]→[-1,1] and the [-1,0] half got clamped to -128.
      input->data.int8[3 * idx + 0] = ch0_buffer[idx];
      // Ch1 & Ch2: flatness/rms are now in [0,1], quantize correctly
      input->data.int8[3 * idx + 1] = flat_q;            // flatness
      input->data.int8[3 * idx + 2] = rms_q;             // RMS
    }
  }

  return SL_STATUS_OK;
}

// ── Route to correct fill function based on tensor size ────────────────────
static sl_status_t fill_input_tensor_from_audio_features(TfLiteTensor* input)
{
  if (input == NULL) return SL_STATUS_NULL_POINTER;

  const int feature_elements = sl_ml_audio_feature_generation_get_feature_buffer_size();
  const int input_elements   = tensor_element_count(input);

  if (feature_elements <= 0 || input_elements <= 0) return SL_STATUS_INVALID_PARAMETER;

  if (input_elements == feature_elements) {
    return sl_ml_audio_feature_generation_fill_tensor(input);
  }

  if (input_elements == (feature_elements * 3)) {
    return fill_input_tensor_3ch_from_features(input, feature_elements);
  }

  if (!input_shape_warned) {
    input_shape_warned = true;
    printf("[ERROR] Input elements=%d do not match frontend=%d (or x3)\n",
           input_elements, feature_elements);
  }
  return SL_STATUS_INVALID_PARAMETER;
}

// ── Debug: print tensor shape ───────────────────────────────────────────────
static void log_tensor_shape(const char* name, const TfLiteTensor* tensor)
{
  if (tensor == NULL || tensor->dims == NULL) {
    printf("[MODEL] %s: unavailable\n", name);
    return;
  }
  printf("[MODEL] %s shape=[", name);
  for (int i = 0; i < tensor->dims->size; i++) {
    printf("%d", tensor->dims->data[i]);
    if (i + 1 < tensor->dims->size) printf(",");
  }
  printf("] bytes=%d type=%d scale=%.6f zp=%ld\n",
         (int)tensor->bytes,
         (int)tensor->type,
         tensor->params.scale,
         (long)tensor->params.zero_point);
}

// ── Run TFLite inference ────────────────────────────────────────────────────
static sl_status_t run_inference(void)
{
  static int      error_count  = 0;
  static uint32_t infer_count  = 0;

  TfLiteTensor* input = sl_tflite_micro_get_input_tensor();
  if (input == NULL) {
    if (error_count++ < 3) printf("[ERROR] Input tensor is NULL\n");
    return SL_STATUS_FAIL;
  }

  sl_status_t status = fill_input_tensor_from_audio_features(input);
  if (status != SL_STATUS_OK) {
    error_count++;
    if (error_count <= 3 || (error_count % 50) == 0)
      printf("[ERROR] Feature fill failed: 0x%lx\n", status);
    return SL_STATUS_FAIL;
  }

  auto interpreter = sl_tflite_micro_get_interpreter();
  if (interpreter == NULL) {
    error_count++;
    if (error_count <= 3 || (error_count % 50) == 0)
      printf("[ERROR] Interpreter is NULL\n");
    return SL_STATUS_FAIL;
  }

  uint32_t t0 = sl_sleeptimer_get_tick_count();
  if (infer_count < 3) printf("[TASK] Invoke start #%lu\n", (unsigned long)(infer_count + 1));

  TfLiteStatus invoke_status = interpreter->Invoke();

  uint32_t t1    = sl_sleeptimer_get_tick_count();
  uint32_t dt_ms = ticks_to_ms(t1 - t0);
  if (infer_count < 3 || (infer_count % 20) == 0)
    printf("[TASK] Invoke done #%lu in %lu ms\n",
           (unsigned long)(infer_count + 1), (unsigned long)dt_ms);
  infer_count++;

  if (invoke_status != kTfLiteOk) {
    error_count++;
    if (error_count <= 3 || (error_count % 50) == 0)
      printf("[ERROR] Invoke failed: %d\n", invoke_status);
    return SL_STATUS_FAIL;
  }

  error_count = 0;
  return SL_STATUS_OK;
}

// ── LED control ─────────────────────────────────────────────────────────────
static void leds_off(void)
{
  sl_led_turn_off(&sl_led_led0);
  sl_led_turn_off(&sl_led_led1);
  sl_led_turn_off(&sl_led_led2);
}

static void set_led_background_blue(void)
{
  // Board has RGB-like discrete LEDs:
  // LED0 is used as blue background indicator when not in cry.
  sl_led_turn_on(&sl_led_led0);
  sl_led_turn_off(&sl_led_led1);
  sl_led_turn_off(&sl_led_led2);
}

static void enter_background_state(void)
{
  set_led_background_blue();
  uint32_t now        = sl_sleeptimer_get_tick_count();
  uint32_t elapsed_ms = ticks_to_ms(now - last_ble_notify_tick);
  if (last_class != BLE_CLASS_BACKGROUND || elapsed_ms >= BLE_NOTIFY_INTERVAL_MS) {
    last_ble_notify_tick = now;
    send_ble_notification(BLE_CLASS_BACKGROUND, 0);
  }
  last_class = BLE_CLASS_BACKGROUND;
}

static void set_leds_for_class(int winner)
{
  if (winner == CLASS_LAUGH) {
    sl_led_turn_on(&sl_led_led0);
    sl_led_turn_on(&sl_led_led1);
    sl_led_turn_off(&sl_led_led2);
  } else if (winner == CLASS_SAD) {
    sl_led_turn_off(&sl_led_led0);
    sl_led_turn_off(&sl_led_led1);
    sl_led_turn_on(&sl_led_led2);
  } else {
    leds_off();
  }
}

// ── Majority ring-buffer (mirrors Python ContinuousOutputEngine) ────────────
static void majority_push(uint8_t class_id)
{
  majority_ring[majority_head] = class_id;
  majority_head = (majority_head + 1) % MAJORITY_WINDOW_SIZE;
  if (majority_count < MAJORITY_WINDOW_SIZE) majority_count++;
}

static int majority_decide(void)
{
  int n_sad   = 0;
  int n_laugh = 0;
  int n       = (majority_count < MAJORITY_WINDOW_SIZE) ? majority_count : MAJORITY_WINDOW_SIZE;

  for (int i = 0; i < n; i++) {
    if (majority_ring[i] == CLASS_SAD)   n_sad++;
    if (majority_ring[i] == CLASS_LAUGH) n_laugh++;
  }

  int n_cry = n_sad + n_laugh;
  if (n_cry < MAJORITY_MIN_CRY_COUNT) return (int)NO_CLASS_ID;
  if (n_sad >= 2 && n_laugh >= 2) return (int)NO_CLASS_ID;
  if ((n_sad > n_laugh ? (n_sad - n_laugh) : (n_laugh - n_sad)) < 2) {
    return (int)NO_CLASS_ID;
  }
  return (n_sad >= n_laugh) ? CLASS_SAD : CLASS_LAUGH;
}

// ── Main output processing — called every 500ms ─────────────────────────────
static void process_output(void)
{
  if (!device_enabled || deep_sleep_enabled || !monitoring_enabled) {
    enter_background_state();
    return;
  }

  TfLiteTensor* output = sl_tflite_micro_get_output_tensor();
  if (output == NULL) return;

  // Verify we have at least 4 output elements
  const int total_out = tensor_element_count(output);
  if (total_out < OUT_TOTAL) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      printf("[WARN] Output tensor has %d elements, expected %d\n", total_out, OUT_TOTAL);
    }
    return;
  }

  // ── Feature variance gate — reject silent frames ──────────────────────
  TfLiteTensor* input = sl_tflite_micro_get_input_tensor();
  if (input != NULL) {
    const int N = (int)input->bytes;
    if (N > 0) {
      int64_t s1 = 0, s2 = 0;
      if (input->type == kTfLiteInt8) {
        const int8_t* feat = input->data.int8;
        for (int i = 0; i < N; i++) { s1 += feat[i]; s2 += (int64_t)feat[i] * feat[i]; }
      } else {
        const int elements = N / (int)sizeof(float);
        const float* feat_f = input->data.f;
        for (int i = 0; i < elements; i++) {
          int32_t q = (int32_t)(feat_f[i] * 128.0f);
          s1 += q; s2 += (int64_t)q * q;
        }
      }
      const int    denom   = (input->type == kTfLiteFloat32)
                             ? (N / (int)sizeof(float)) : N;
      const int32_t fmean  = (int32_t)(s1 / denom);
      const int32_t feat_var = (int32_t)(s2 / denom) - fmean * fmean;

      if (feat_var < FEATURE_VARIANCE_THRESHOLD) {
        printf("UNCERTAIN(var=%d) -> BG\n", (int)feat_var);
        majority_push(NO_CLASS_ID);
        enter_background_state();
        return;
      }
    }
  }

  // ── Dequantize all 4 outputs to float [0.0, 1.0] ─────────────────────
  // This is the fix — raw int8 values are NOT directly comparable,
  // they must be dequantized using the tensor's scale and zero_point.
  const float p_cry     = dequant_output(output, OUT_IDX_CRY);
  const float p_not_cry = dequant_output(output, OUT_IDX_NOT_CRY);
  const float p_sad     = dequant_output(output, OUT_IDX_SAD);
  const float p_laugh   = dequant_output(output, OUT_IDX_LAUGH);

  printf("p_cry=%.2f p_not_cry=%.2f p_sad=%.2f p_laugh=%.2f | ",
         p_cry, p_not_cry, p_sad, p_laugh);

  // ── Stage 1 gate with hysteresis — is there a cry at all? ────────────
  const bool cry_enter = (p_cry >= CRY_CONF_ENTER_THRESHOLD)
                         && (p_cry >= (p_not_cry + CRY_OVER_NOT_CRY_MARGIN));
  const bool cry_keep = (p_cry >= CRY_CONF_EXIT_THRESHOLD)
                        && (p_cry >= (p_not_cry + CRY_OVER_NOT_CRY_MARGIN));

  if (cry_enter) {
    cry_gate_open = true;
    non_cry_streak = 0;
  } else if (cry_gate_open && cry_keep) {
    non_cry_streak = 0;
  } else {
    if (non_cry_streak < 255u) non_cry_streak++;
    if (non_cry_streak >= NON_CRY_CLOSE_FRAMES) {
      cry_gate_open = false;
      non_cry_streak = NON_CRY_CLOSE_FRAMES;
    }
  }

  if (!cry_gate_open) {
    printf("GATE_BLOCK -> BG\n");
    majority_push(NO_CLASS_ID);
    enter_background_state();
    return;
  }

  // ── Stage 2 — within cry, choose emotion by sad vs laugh head ─────────
  const int raw_class = (p_sad >= p_laugh) ? CLASS_SAD : CLASS_LAUGH;
  float cry_conf_pct_f = p_cry * 100.0f;
  if (cry_conf_pct_f < 0.0f) cry_conf_pct_f = 0.0f;
  if (cry_conf_pct_f > 100.0f) cry_conf_pct_f = 100.0f;
  const uint8_t report_score = (uint8_t)(cry_conf_pct_f + 0.5f);

  if (report_score < app_conf_threshold) {
    majority_push(NO_CLASS_ID);
    enter_background_state();
    return;
  }

  // ── Push into majority ring ───────────────────────────────────────────
  majority_push((uint8_t)raw_class);
  const int winner = majority_decide();

  printf("raw=%s | ring: ", CLASS_NAMES[raw_class]);
  {
    int n = (majority_count < MAJORITY_WINDOW_SIZE) ? majority_count : MAJORITY_WINDOW_SIZE;
    for (int i = 0; i < n; i++) {
      uint8_t v = majority_ring[i];
      printf("%s ", (v == CLASS_LAUGH) ? "L" : (v == CLASS_SAD) ? "S" : ".");
    }
  }

  if (winner == (int)NO_CLASS_ID) {
    printf("-> BG\n");
    enter_background_state();
    return;
  }

  printf("-> %s\n", CLASS_NAMES[winner]);

  // ── BLE notification ──────────────────────────────────────────────────
  // BLE contract: 0=BACKGROUND, 1=LAUGH, 2=SAD
  const uint8_t report_class = (winner == CLASS_SAD) ? BLE_CLASS_SAD : BLE_CLASS_LAUGH;
  uint32_t now        = sl_sleeptimer_get_tick_count();
  uint32_t elapsed_ms = ticks_to_ms(now - last_ble_notify_tick);
  if (elapsed_ms >= BLE_NOTIFY_INTERVAL_MS || last_class != report_class) {
    last_ble_notify_tick = now;
    send_ble_notification(report_class, report_score);
  }
  last_class = report_class;

  // ── LED ───────────────────────────────────────────────────────────────
  set_leds_for_class(winner);
}

// ── Init ────────────────────────────────────────────────────────────────────
void audio_classifier_init(void)
{
  RTOS_ERR err;
  OSTaskCreate(&tcb,
               (CPU_CHAR *)"audio task",
               audio_classifier_task,
               DEF_NULL,
               TASK_PRIORITY,
               &stack[0],
               (TASK_STACK_SIZE / 10u),
               TASK_STACK_SIZE,
               0u,
               0u,
               DEF_NULL,
               (OS_OPT_TASK_STK_CLR),
               &err);
  EFM_ASSERT((RTOS_ERR_CODE_GET(err) == RTOS_ERR_NONE));
}

// ── Main task loop ──────────────────────────────────────────────────────────
void audio_classifier_task(void *arg)
{
  RTOS_ERR err;
  (void)arg;

  printf("\n=== AUDIO CLASSIFIER v10 (2-Class Laugh/Sad + CryConf Gate) ===\n");
  printf("Output layout : [P(CRY), P(NOT_CRY), P(SAD), P(LAUGH)]\n");
  printf("Stage1 gate   : enter>=%.2f, exit<%.2f\n",
         CRY_CONF_ENTER_THRESHOLD, CRY_CONF_EXIT_THRESHOLD);
  printf("Majority win  : %d of %d steps\n", MAJORITY_MIN_CRY_COUNT, MAJORITY_WINDOW_SIZE);
  printf("==============================================================\n\n");

  printf("[TASK] Sleeptimer freq: %lu Hz\n",
         (unsigned long)sl_sleeptimer_get_timer_frequency());

  printf("[TASK] Enabling microphone sensor...\n");
  sl_status_t mic_status = sl_board_enable_sensor(SL_BOARD_SENSOR_MICROPHONE);
  if (mic_status == SL_STATUS_OK) {
    printf("[TASK] Microphone enabled OK\n");
  } else {
    printf("[ERROR] Microphone enable failed: 0x%lx\n", mic_status);
  }

  OSTimeDlyHMSM(0, 0, 0, 100, OS_OPT_TIME_HMSM_NON_STRICT, &err);

  printf("[TASK] Initializing audio feature generation...\n");
  sl_ml_audio_feature_generation_init();
  printf("[TASK] Audio init complete\n");

  sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);
  em1_requirement_held = true;

  printf("[TASK] Waiting for audio buffer to fill (2 seconds)...\n");
  OSTimeDlyHMSM(0, 0, 2, 0, OS_OPT_TIME_HMSM_NON_STRICT, &err);
  printf("[TASK] Buffer fill complete\n");

  // Log tensor shapes + quantization params at startup
  log_tensor_shape("input",  sl_tflite_micro_get_input_tensor());
  log_tensor_shape("output", sl_tflite_micro_get_output_tensor());

  {
    TfLiteTensor* input           = sl_tflite_micro_get_input_tensor();
    const int     feature_elements = sl_ml_audio_feature_generation_get_feature_buffer_size();
    const int     input_elements   = tensor_element_count(input);
    if (feature_elements > 0 && input_elements > 0) {
      if (input_elements == feature_elements * 3) {
        printf("[INFO] 3-channel mode: mel + flatness + RMS\n");
      } else if (input_elements == feature_elements) {
        printf("[INFO] 1-channel mode: mel only\n");
      } else {
        printf("[ERROR] Input elements=%d do not match frontend=%d (or x3)\n",
               input_elements, feature_elements);
      }
    }
  }

  // Init majority ring to background
  for (int i = 0; i < MAJORITY_WINDOW_SIZE; i++) {
    majority_ring[i] = NO_CLASS_ID;
  }

  printf("[TASK] Starting inference loop (500ms interval)...\n");
  int  loop_count   = 0;
  bool first_success = false;

  while (1) {
    app_service_temperature_telemetry();

    if (!device_enabled) {
      if (em1_requirement_held) {
        sl_power_manager_remove_em_requirement(SL_POWER_MANAGER_EM1);
        em1_requirement_held = false;
      }
      leds_off();
      enter_background_state();
      OSTimeDlyHMSM(0, 0, 0, CONTROL_IDLE_INTERVAL_MS, OS_OPT_TIME_HMSM_NON_STRICT, &err);
      continue;
    } else if (deep_sleep_enabled) {
      if (em1_requirement_held) {
        sl_power_manager_remove_em_requirement(SL_POWER_MANAGER_EM1);
        em1_requirement_held = false;
      }
      leds_off();
      enter_background_state();
      OSTimeDlyHMSM(0, 0, 0, DEEP_SLEEP_IDLE_INTERVAL_MS, OS_OPT_TIME_HMSM_NON_STRICT, &err);
      continue;
    } else if (!em1_requirement_held) {
      sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);
      em1_requirement_held = true;
    }

    OSTimeDlyHMSM(0, 0, 0, INFERENCE_INTERVAL_MS, OS_OPT_TIME_PERIODIC, &err);
    sl_ml_audio_feature_generation_update_features();

    if (run_inference() == SL_STATUS_OK) {
      if (!first_success) {
        printf("[TASK] First inference SUCCESS — classifier running.\n");
        first_success = true;
      }
      process_output();
    } else if (loop_count < 10 || (loop_count % 50) == 0) {
      printf("[TASK] Loop %d: inference not ready yet...\n", loop_count);
    }

    loop_count++;
  }
}
