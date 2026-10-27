/***************************************************************************//**
 * @file
 * @brief Core application logic.
 *******************************************************************************
 * # License
 * <b>Copyright 2020 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * SPDX-License-Identifier: Zlib
 *
 * The licensor of this software is Silicon Laboratories Inc.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 ******************************************************************************/

#include "bluetooth.h"

static uint8_t input_report_data[9];

static uint8_t notification_enabled = 0;
static uint8_t connection_handle = 0xff;
// True while a key press report has been sent and the matching release has not.
static bool key_report_pending = false;
uint8_t current_report_page = 0; // 0x01 is for vanilla keyboard report, 0x02 is for media keys
uint8_t current_key = 0;
uint8_t current_modifier_key = 0;

// The advertising set handle allocated from Bluetooth stack.
static uint8_t advertising_set_handle = 0xff;
void print_report(uint8_t * report, uint8_t len)
{
  for (uint8_t i = 0; i < len; i++) {
    app_log("%02X ", report[i]);
  }
  app_log("\n");
}
/**************************************************************************//**
 * Bluetooth stack event handler.
 * This overrides the dummy weak implementation.
 *
 * @param[in] evt Event coming from the Bluetooth stack.
 *****************************************************************************/
void sl_bt_on_event(sl_bt_msg_t *evt)
{
  sl_status_t sc;
  bd_addr address;
  uint8_t address_type;
  uint8_t system_id[8];
  switch (SL_BT_MSG_ID(evt->header)) {
    // -------------------------------
    // This event indicates the device has started and the radio is ready.
    // Do not call any stack command before receiving this boot event!
    case sl_bt_evt_system_boot_id:
      app_log_info("Bluetooth stack booted: v%d.%d.%d-b%d\n",
                   evt->data.evt_system_boot.major,
                   evt->data.evt_system_boot.minor,
                   evt->data.evt_system_boot.patch,
                   evt->data.evt_system_boot.build);
      // Extract unique ID from BT Address.
      sc = sl_bt_system_get_identity_address(&address, &address_type);
      app_assert_status(sc);

      // Pad and reverse unique ID to get System ID.
      system_id[0] = address.addr[5];
      system_id[1] = address.addr[4];
      system_id[2] = address.addr[3];
      system_id[3] = 0xFF;
      system_id[4] = 0xFE;
      system_id[5] = address.addr[2];
      system_id[6] = address.addr[1];
      system_id[7] = address.addr[0];

      sc = sl_bt_gatt_server_write_attribute_value(gattdb_system_id,
                                                   0,
                                                   sizeof(system_id),
                                                   system_id);
      app_assert_status(sc);

      app_log_info("Bluetooth %s address: %02X:%02X:%02X:%02X:%02X:%02X\n",
                   address_type ? "static random" : "public device",
                   address.addr[5],
                   address.addr[4],
                   address.addr[3],
                   address.addr[2],
                   address.addr[1],
                   address.addr[0]);

      sl_bt_sm_set_bondable_mode(1);
      // Create an advertising set.
      sc = sl_bt_advertiser_create_set(&advertising_set_handle);
      app_assert_status(sc);

      // Generate data for advertising
      sc = sl_bt_legacy_advertiser_generate_data(advertising_set_handle,
                                                 sl_bt_advertiser_general_discoverable);
      app_assert_status(sc);

      // Set advertising interval to 100ms.
      sc = sl_bt_advertiser_set_timing(
        advertising_set_handle,
        160, // min. adv. interval (milliseconds * 1.6)
        160, // max. adv. interval (milliseconds * 1.6)
        0,   // adv. duration
        0);  // max. num. adv. events
      app_assert_status(sc);

      // Start advertising and enable connections.
      sc = sl_bt_legacy_advertiser_start(advertising_set_handle,
                                         sl_bt_legacy_advertiser_connectable);
      app_assert_status(sc);
      app_log_info("Started advertising\n");
      break;
    // -------------------------------
    // This event indicates that a new connection was opened.
    case sl_bt_evt_connection_opened_id:
      notification_enabled = 0;
      key_report_pending = false;
      connection_handle = evt->data.evt_connection_opened.connection;
      sl_bt_sm_increase_security(evt->data.evt_connection_opened.connection);
      app_log_info("Connection opened.\n");
      break;

    // -------------------------------
    // This event indicates that a connection was closed.
    case sl_bt_evt_connection_closed_id:
      notification_enabled = 0;
      key_report_pending = false;
      connection_handle = 0xff;
      app_log_info("Connection closed.\n");

      // Generate data for advertising
      sc = sl_bt_legacy_advertiser_generate_data(advertising_set_handle,
                                                 sl_bt_advertiser_general_discoverable);
      app_assert_status(sc);

      // Restart advertising after client has disconnected.
      sc = sl_bt_legacy_advertiser_start(advertising_set_handle,
                                         sl_bt_legacy_advertiser_connectable);
      app_assert_status(sc);
      break;
    case sl_bt_evt_gatt_server_characteristic_status_id:
      app_log_info("Characteristic was attempted changed: %d\n", evt->data.evt_gatt_server_characteristic_status.characteristic);
      if (evt->data.evt_gatt_server_characteristic_status.characteristic == gattdb_report) {
        // A local Client Characteristic Configuration descriptor was changed in
        // the gattdb_report characteristic.
        if (evt->data.evt_gatt_server_characteristic_status.client_config_flags
            & sl_bt_gatt_notification) {
          // The client just enabled the notification.
          notification_enabled = 1;
          app_log_info("Notification enabled.\n");
        } else {
          notification_enabled = 0;
          app_log_info("Notification disabled.\n");
        }
      }
      break;
    case sl_bt_evt_sm_bonding_failed_id:
      app_log_error("Bonding failed, reason: 0x%04X\n", (unsigned int)evt->data.evt_sm_bonding_failed.reason);
      sl_bt_sm_delete_bondings();
      break;
    ///////////////////////////////////////////////////////////////////////////
    // Add additional event handlers here as your application requires!      //
    ///////////////////////////////////////////////////////////////////////////
    case  sl_bt_evt_system_external_signal_id:
      if ((notification_enabled == 1) && (connection_handle != 0xff)) {
        uint8_t len = 0;
        memset(input_report_data, 0, sizeof(input_report_data));
        input_report_data[0] = current_report_page;
        if (current_report_page == REPORT_PAGE_KEYBOARD) {
          len = 9;
          input_report_data[3] = current_key;
        } else if (current_report_page == REPORT_PAGE_MEDIA) {
          len = 2;
          input_report_data[1] = current_key;
        }

        sc = sl_bt_gatt_server_send_notification(connection_handle, gattdb_report, len, input_report_data);

        app_assert_status(sc);
        key_report_pending = true;
        app_log_info("Key report was sent: ");
        print_report(input_report_data, len);
      }
      break;
    // -------------------------------
    // Default event handler.
    default:
      app_log("0x%08lX\n", (unsigned long)SL_BT_MSG_ID(evt->header));
      break;
  }
}

void reset_report()
{
  // Only release a key that was actually pressed.
  if (!key_report_pending) {
    return;
  }
  key_report_pending = false;
  if ((notification_enabled == 1) && (connection_handle != 0xff)) {
    memset(input_report_data, 0, sizeof(input_report_data));
    input_report_data[0] = current_report_page;
    uint8_t len = 0;
    if (current_report_page == REPORT_PAGE_KEYBOARD) {
      len = 9;
    } else if (current_report_page == REPORT_PAGE_MEDIA) {
      len = 2;
    }
    sl_status_t sc = sl_bt_gatt_server_send_notification(connection_handle, gattdb_report, len, input_report_data);
    app_assert_status(sc);
    app_log_info("Key report was reset: ");
    print_report(input_report_data, len);
  }
}
