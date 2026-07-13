#include "bluetooth.h"
#include <stdio.h>
#include <string.h>

static uint8_t notification_enabled = 0;
static uint8_t connection_handle = 0xff;

// The advertising set handle allocated from Bluetooth stack.
static uint8_t advertising_set_handle = 0xff;

void assert_status(sl_status_t sc)
{
  if (sc != SL_STATUS_OK)
  {
    app_log("Error occurred: 0x%04lX\n", (unsigned long)sc);
  }
}

void start_advertising()
{
  sl_status_t sc;

  // Generate data for advertising
  sc = sl_bt_legacy_advertiser_generate_data(advertising_set_handle,
                                             sl_bt_advertiser_general_discoverable);
  assert_status(sc);

  // Start advertising and enable connections.
  sc = sl_bt_legacy_advertiser_start(advertising_set_handle,
                                     sl_bt_advertiser_connectable_scannable);
  assert_status(sc);

  app_log_info("Started advertising\n");
}

/**************************************************************************/
/**
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

  switch (SL_BT_MSG_ID(evt->header))
  {
  case sl_bt_evt_system_boot_id:
    app_log_info("Bluetooth stack booted: v%d.%d.%d-b%d\n",
                 evt->data.evt_system_boot.major,
                 evt->data.evt_system_boot.minor,
                 evt->data.evt_system_boot.patch,
                 evt->data.evt_system_boot.build);

    sc = sl_bt_system_get_identity_address(&address, &address_type);
    assert_status(sc);

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
    assert_status(sc);

    app_log_info("Bluetooth %s address: %02X:%02X:%02X:%02X:%02X:%02X\n",
                 address_type ? "static random" : "public device",
                 address.addr[5],
                 address.addr[4],
                 address.addr[3],
                 address.addr[2],
                 address.addr[1],
                 address.addr[0]);

    sc = sl_bt_sm_set_bondable_mode(0);
    assert_status(sc);

    sc = sl_bt_advertiser_create_set(&advertising_set_handle);
    assert_status(sc);

    sc = sl_bt_advertiser_set_timing(advertising_set_handle,
                                     160,
                                     160,
                                     0,
                                     0);
    assert_status(sc);

    start_advertising();
    break;

  case sl_bt_evt_connection_opened_id:
    notification_enabled = 0;
    connection_handle = evt->data.evt_connection_opened.connection;
    app_log_info("Connection opened.\n");
    break;

  case sl_bt_evt_connection_closed_id:
    notification_enabled = 0;
    connection_handle = 0xff;
    app_log_info("Connection closed.\n");
    start_advertising();
    break;

  case sl_bt_evt_sm_bonding_failed_id:
    app_log("Bonding failed: 0x%04lX\n",
            (unsigned long)evt->data.evt_sm_bonding_failed.reason);
    break;

  case sl_bt_evt_gatt_server_characteristic_status_id:
    app_log_info("Characteristic status changed: %lu\n",
                 (unsigned long)evt->data.evt_gatt_server_characteristic_status.characteristic);

    if (evt->data.evt_gatt_server_characteristic_status.characteristic == gattdb_data_capture_data)
    {
      if (evt->data.evt_gatt_server_characteristic_status.client_config_flags & sl_bt_gatt_notification)
      {
        notification_enabled = 1;
        app_log_info("Notification enabled.\n");
      }
      else
      {
        notification_enabled = 0;
        app_log_info("Notification disabled.\n");
      }
    }
    break;

  default:
    app_log("0x%08lX\n", (unsigned long)SL_BT_MSG_ID(evt->header));
    break;
  }
}

#define PACKET_SIZE 242

static uint8_t packet[PACKET_SIZE];
static size_t remaining_packet = PACKET_SIZE;

void data_notify(void *buf_ptr, size_t len)
{
  const uint8_t *src = (const uint8_t *)buf_ptr;
  sl_status_t sc;
  size_t remaining_len = len;

  if (connection_handle == 0xff || notification_enabled == 0)
  {
    return;
  }

  while (remaining_len > 0)
  {
    size_t len_to_send = remaining_packet;
    if (remaining_len < len_to_send)
    {
      len_to_send = remaining_len;
    }

    memcpy(packet + (PACKET_SIZE - remaining_packet),
           src + (len - remaining_len),
           len_to_send);

    remaining_packet -= len_to_send;
    remaining_len -= len_to_send;

    if (remaining_packet == 0)
    {
      sc = sl_bt_gatt_server_send_notification(connection_handle,
                                               gattdb_data_capture_data,
                                               PACKET_SIZE,
                                               packet);
      if (sc != SL_STATUS_OK)
      {
        app_log_error("[E: 0x%04lX] Failed to send characteristic notification\n",
                      (unsigned long)sc);
      }

      remaining_packet = PACKET_SIZE;
      sl_sleeptimer_delay_millisecond(10);
    }
  }
}