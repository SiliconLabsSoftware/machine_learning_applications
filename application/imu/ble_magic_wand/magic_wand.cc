/***************************************************************************//**
 * @file
 * @brief Initalization and main functionality for application
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/
#include "sl_ml_model_magic_wand_model.h"
#include "sl_sleeptimer.h"
#include "magic_wand.h"
#include "accelerometer.h"
#include "sl_simple_button_instances.h"
#include "math.h"
static int input_length;
static TfLiteTensor* model_input;
static acc_data_t buf[SEQUENCE_LENGTH] = { 0.5f, 0.5f, 0.5f };
static bool infer = false;
static bool read_accel = false;
// Set from the button interrupt; the Bluetooth stack must not be called from
// interrupt context, so the key release report is sent from magic_wand_action().
static volatile bool report_reset_pending = false;
static int head_ptr = 0;
static int inference_trigger_samples_num = round(INFERENCE_PERIOD_MS / ACCELEROMETER_FREQ);
static acc_data_t prev_data = { 0.5f, 0.5f, 0.5f };
static void listen_for_gestures(bool enable)
{
  if (enable) {
    for (uint8_t i = 0; i < SEQUENCE_LENGTH; i++) {
      acc_data_t _d = { 0.5f, 0.5f, 0.5f };
      buf[i] = _d;
    }
    read_accel = true;
  } else {
    read_accel = false;
    head_ptr = 0;
  }
}
void sl_button_on_change(const sl_button_t *handle)
{
  if (sl_button_get_state(handle) == SL_SIMPLE_BUTTON_PRESSED) {
    if (&sl_button_btn0 == handle) {
      listen_for_gestures(true);
    }
  } else if (sl_button_get_state(handle) == SL_SIMPLE_BUTTON_RELEASED) {
    if (&sl_button_btn0 == handle) {
      listen_for_gestures(false);
      report_reset_pending = true;
    }
  }
}

// Called when the IMU has data available using gpio interrupt.
static void on_data_available(uint8_t int_id, void *ctx)
{
  (void) int_id;
  (void) ctx;
  acc_data_t data = { 0, 0, 0 };
  sl_status_t status = accelerometer_read(&data);
  if (status == SL_STATUS_FAIL || !read_accel) {
    return;
  }
  // Range: [-2000, 2000]
  data.x /= 2000;
  data.y /= 2000;
  data.z /= 2000;
  // Range: [-1, 1]
  acc_data_t delta_data = { 0 };
  delta_data.x = data.x - prev_data.x;
  delta_data.y = data.y - prev_data.y;
  delta_data.z = data.z - prev_data.z;
  // Range: [-2, 2]
  delta_data.x = (delta_data.x / 2 + 1) / 2;
  delta_data.y = (delta_data.y / 2 + 1) / 2;
  delta_data.z = (delta_data.z / 2 + 1) / 2;
  // Range: [0, 1]
  buf[head_ptr].x = delta_data.x;
  buf[head_ptr].y = delta_data.y;
  buf[head_ptr].z = delta_data.z;

  head_ptr++;
  prev_data.x = data.x;
  prev_data.y = data.y;
  prev_data.z = data.z;
  if (head_ptr >= SEQUENCE_LENGTH) {
    head_ptr = 0;
  }
  if (head_ptr % inference_trigger_samples_num == 0) {
    infer = true;
  }
}
void magic_wand_init(void)
{
  app_log_info("Initializing Magic Wand...\n");
  // Obtain pointer to the model's input tensor.
  if (sl_ml_model_init(&sl_ml_magic_wand_model_model_handle) != SL_STATUS_OK) {
    app_log_error("Failed to load model.\n");
    EFM_ASSERT(false);
    return;
  }
  model_input = sl_ml_magic_wand_model_model_handle.input_tensor(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1)
      || (model_input->dims->data[1] != SEQUENCE_LENGTH)
      || (model_input->dims->data[2] != ACCELEROMETER_CHANNELS)
      || (model_input->type != kTfLiteFloat32)) {
    app_log_error("Bad input tensor parameters in model.\n");
    EFM_ASSERT(false);
    return;
  }

  input_length = model_input->bytes / sizeof(float);

  sl_status_t setup_status = accelerometer_setup(on_data_available);

  if (setup_status != SL_STATUS_OK) {
    app_log_error("Accelerometer setup failed.\n");
    EFM_ASSERT(false);
    return;
  }

  // Waiting for accelerometer to become ready
  while (true) {
    sl_status_t status = accelerometer_read(NULL);
    if (status == SL_STATUS_OK) {
      break;
    }
  }
  app_log_info("Ready.\n");
}

static void handle_output(int gesture)
{
  if (gesture == GESTURE_UP) {
    app_log_info("up\n");
    current_report_page = REPORT_PAGE_KEYBOARD;
    current_key = KEY_ARROW_UP;
    sl_bt_external_signal(1);
    // We dont want multiple of this gesture, so we stop listening
    listen_for_gestures(false);
  } else if (gesture == GESTURE_CIRCLE) {
    app_log_info("circle\n");
    current_report_page = REPORT_PAGE_MEDIA;
    current_key = KEY_MEDIA_PLAYPAUSE;
    sl_bt_external_signal(1);
    // We dont want multiple of this gesture, so we stop listening
    listen_for_gestures(false);
  } else if (gesture == GESTURE_DOWN) {
    app_log_info("down\n");
    current_report_page = REPORT_PAGE_KEYBOARD;
    current_key = KEY_ARROW_DOWN;
    sl_bt_external_signal(1);
    // We dont want multiple of this gesture, so we stop listening
    listen_for_gestures(false);
  } else if (gesture == GESTURE_UNKNOWN) {
    // Unknown gesture, do nothing
    return;
  }
}
static void do_inference(void)
{
  infer = false;
  // Copy the buffer into the model inputs
  memcpy(model_input->data.f, buf, sizeof(buf));

  if (sl_ml_model_run(&sl_ml_magic_wand_model_model_handle) != SL_STATUS_OK) {
    app_log_error("Inference failed!\n");
    return;
  }

  // Analyze the results to obtain a prediction
  const model_output_t *output = (const model_output_t *)sl_ml_magic_wand_model_model_handle.output_tensor(0)->data.f;
  int gesture = GESTURE_UNKNOWN;
#ifdef MAGIC_WAND_LOG_SCORES
  if ((*output).gesture[0] > 0.3f || (*output).gesture[1] > 0.3f || (*output).gesture[2] > 0.3f) {
    app_log_info("scores down=%d circle=%d up=%d unknown=%d (x1000)\n",
                 (int)((*output).gesture[0] * 1000), (int)((*output).gesture[1] * 1000),
                 (int)((*output).gesture[2] * 1000), (int)((*output).gesture[3] * 1000));
  }
#endif
  if ((*output).gesture[0] > DETECTION_THRESHOLD) {
    gesture = GESTURE_DOWN;
  } else if ((*output).gesture[1] > DETECTION_THRESHOLD) {
    gesture = GESTURE_CIRCLE;
  } else if ((*output).gesture[2] > DETECTION_THRESHOLD) {
    gesture = GESTURE_UP;
  }
  // Produce an output
  handle_output(gesture);
}
void magic_wand_action(void)
{
  if (report_reset_pending) {
    report_reset_pending = false;
    reset_report();
  }
  if (infer) {
    do_inference();
  }
}
