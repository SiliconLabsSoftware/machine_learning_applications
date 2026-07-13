/***************************************************************************//**
 * @file app.h
 * @brief Top level application functions + BLE shared state
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

#ifndef APP_H
#define APP_H

#include <stdbool.h>
#include <stdint.h>

/***************************************************************************//**
 * Initialize application.
 ******************************************************************************/
void app_init(void);

/***************************************************************************//**
 * App tance action. Called in the main loop.
 ******************************************************************************/
void app_process_action(void);

// ============================================================================
// BLE SHARED STATE (read/write from audio_classifier.cc)
// ============================================================================
extern volatile bool notifications_enabled;  // true when phone enables CCCD
extern uint8_t       connection_handle;      // 0xFF = not connected
extern uint8_t       last_class;             // last notified class (0xFF = none)
extern volatile bool device_enabled;         // true when device monitoring is enabled
extern volatile bool monitoring_enabled;     // true when inference processing is enabled
extern volatile bool deep_sleep_enabled;     // true when EM2 deep sleep mode is requested
extern volatile uint8_t app_conf_threshold;  // confidence threshold sent by app (0..100)
extern volatile uint8_t app_debounce_count;  // debounce count sent by app (1..10)

// ============================================================================
// BLE NOTIFICATION FUNCTION (called from audio_classifier.cc)
// ============================================================================
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Send BLE notification with inference result
 * @param class_id 0=background, 1=laugh, 2=sad
 * @param confidence_pct 0..100 (typically cry_conf * 100)
 */
void send_ble_notification(uint8_t class_id, uint8_t confidence_pct);
void app_apply_control_command(uint8_t command_id, uint8_t value);
void app_service_temperature_telemetry(void);

#ifdef __cplusplus
}
#endif

#endif  // APP_H
