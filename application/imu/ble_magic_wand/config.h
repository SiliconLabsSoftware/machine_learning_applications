/***************************************************************************//**
 * @file
 * @brief Config for use in the sample application
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

#ifndef MAGIC_WAND_CONFIG_H
#define MAGIC_WAND_CONFIG_H

// The expected accelerometer data sample frequency
#define ACCELEROMETER_FREQ      25
#define ACCELEROMETER_CHANNELS   3

// Inference it triggered by a periodic timer, this configuration is the time
// between each trigger
#define INFERENCE_PERIOD_MS    100

// Length of the accelerator input sequence expected by the model
#define SEQUENCE_LENGTH         50

// What gestures are supported.
#define GESTURE_COUNT            4
#define GESTURE_DOWN             0
#define GESTURE_CIRCLE           1
#define GESTURE_UP               2
#define GESTURE_UNKNOWN          3

// These control the sensitivity of the detection algorithm. If you're seeing
// too many false positives or not enough true positives, you can try tweaking
// these thresholds. Often, increasing the size of the training set will give
// more robust results though, so consider retraining if you are seeing poor
// predictions.
#define DETECTION_THRESHOLD   0.9f
#define PREDICTION_HISTORY_LEN   5
#define PREDICTION_SUPPRESSION  18

// Uncomment to log the model output scores on every inference while holding
// btn0. Useful when tuning DETECTION_THRESHOLD or checking a retrained model.
// #define MAGIC_WAND_LOG_SCORES

#endif // MAGIC_WAND_CONFIG_H
