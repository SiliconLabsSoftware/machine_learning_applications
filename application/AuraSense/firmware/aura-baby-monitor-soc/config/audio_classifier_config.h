/***************************************************************************//**
 * @file
 * @brief Audio classifier application config
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc.  Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement.  This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/

#ifndef AUDIO_CLASSIFIER_CONFIG_H
#define AUDIO_CLASSIFIER_CONFIG_H

#if __has_include("sl_tflite_micro_model_parameters.h")
  #include "sl_tflite_micro_model_parameters.h"
#endif

// <<< Use Configuration Wizard in Context Menu >>>
// <h> Audio Classification configuration

// ── These smoothing/window settings are kept for SDK compatibility but the
//    actual debounce logic in audio_classifier.cc uses the majority ring
//    buffer (MAJORITY_WINDOW_SIZE / MAJORITY_MIN_CRY_COUNT) instead. ────────

// <o SMOOTHING_WINDOW_DURATION_MS> Smoothing window duration [ms] <300-1000>
// <i> Default: 600
#if defined(SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS)
  #define SMOOTHING_WINDOW_DURATION_MS SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS
#else
  #define SMOOTHING_WINDOW_DURATION_MS 600
#endif

// <o MINIMUM_DETECTION_COUNT> Minimum detection count <0-50>
// <i> Default: 3
#if defined(SL_TFLITE_MODEL_MINIMUM_COUNT)
  #define MINIMUM_DETECTION_COUNT SL_TFLITE_MODEL_MINIMUM_COUNT
#else
  #define MINIMUM_DETECTION_COUNT 3
#endif

// <o DETECTION_THRESHOLD> Detection Threshold <0-255>
// <i> Not used directly — thresholding is done via CRY_CONF_THRESHOLD
// <i> (float 0.0-1.0) in audio_classifier.cc.
// <i> Default: 100
#if defined(SL_TFLITE_MODEL_DETECTION_THRESHOLD)
  #define DETECTION_THRESHOLD SL_TFLITE_MODEL_DETECTION_THRESHOLD
#else
  #define DETECTION_THRESHOLD 100
#endif

// <o SUPPRESSION_TIME_MS> Suppression time after detection [ms] <0-2000>
// <i> Default: 1000
#if defined(SL_TFLITE_MODEL_SUPPRESSION_MS)
  #define SUPPRESSION_TIME_MS SL_TFLITE_MODEL_SUPPRESSION_MS
#else
  #define SUPPRESSION_TIME_MS 1000
#endif

// <o SENSITIVITY> Sensitivity of the activity indicator
// <i> Default: 0.5
#define SENSITIVITY .5f

// <q IGNORE_UNDERSCORE_LABELS> Ignore labels with leading underscore
// <i> Default: 1
#define IGNORE_UNDERSCORE_LABELS 1

// <o DETECTION_LED> LED to use for detection
// <i> Default: sl_led_led1
#define DETECTION_LED sl_led_led1

// <o ACTIVITY_LED> LED to use for activity
// <i> Default: sl_led_led0
#define ACTIVITY_LED sl_led_led0

// <q VERBOSE_MODEL_OUTPUT_LOGS> Enable verbose model output logging
// <i> Default: 1
#define VERBOSE_MODEL_OUTPUT_LOGS 1

// <o INFERENCE_INTERVAL_MS> Delay between each inference [ms]
// <i> Must match HOP_SECONDS (0.5s) from the Python training pipeline.
// <i> Default: 500
#define INFERENCE_INTERVAL_MS 500

// <o MAX_CATEGORY_COUNT> Max number of categories supported.
// <i> Default: 16
#define MAX_CATEGORY_COUNT    16

// <o MAX_RESULT_COUNT> Max number of results supported.
// <i> Default: 50
#define MAX_RESULT_COUNT      50

// <o TASK_STACK_SIZE> Application task stack size.
// <i> Default: 512
#define TASK_STACK_SIZE      512

// <o TASK_PRIORITY> Application task priority.
// <i> Default: 20
#define TASK_PRIORITY         20

// <o CATEGORY_LABELS> Label for each category.
// <i> Class 0 = laugh, Class 1 = sad.
// <i> Must match the exported TFLite model output[0:2] ordering exactly.
// <i> output[0] = laugh score, output[1] = sad score.
#if defined(SL_TFLITE_MODEL_CLASSES)
  #define CATEGORY_LABELS SL_TFLITE_MODEL_CLASSES
#else
  #define CATEGORY_LABELS { "laugh", "sad" }
#endif

// <<< end of configuration section >>>

#endif // AUDIO_CLASSIFIER_CONFIG_H
