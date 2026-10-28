package com.siliconlabs.bledemo.features.demo.babycry

import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.content.Context
import android.content.Intent
import android.media.Ringtone
import android.media.RingtoneManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.format.DateFormat
import android.view.LayoutInflater
import android.view.MenuItem
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.siliconlabs.bledemo.R
import com.siliconlabs.bledemo.base.activities.BaseDemoActivity
import com.siliconlabs.bledemo.bluetooth.ble.GattCharacteristic
import com.siliconlabs.bledemo.bluetooth.ble.GattService
import com.siliconlabs.bledemo.bluetooth.ble.TimeoutGattCallback
import com.siliconlabs.bledemo.databinding.ActivityBabyCryMonitorBinding
import com.siliconlabs.bledemo.features.demo.health_thermometer.models.TemperatureReading
import com.siliconlabs.bledemo.home_screen.dialogs.SelectDeviceDialog
import com.siliconlabs.bledemo.utils.AppUtil
import com.siliconlabs.bledemo.utils.BLEUtils.getCharacteristic
import com.siliconlabs.bledemo.utils.BLEUtils.setNotificationForCharacteristic
import com.siliconlabs.bledemo.utils.CustomToastManager
import com.siliconlabs.bledemo.utils.Notifications
import kotlin.math.max
import kotlin.math.min

@SuppressLint("MissingPermission")
class BabyCryMonitorActivity : BaseDemoActivity() {
    private lateinit var binding: ActivityBabyCryMonitorBinding

    private var serviceHasBeenSet = false
    private var monitoringEnabled = true
    private var powerSavingModeEnabled = false
    private var alertsEnabled = true
    private var confidenceThreshold = 85
    private var debounceCount = 5
    private var escalationThreshold = 5
    private var sadDetectionsInRow = 0
    private var ignoredAlerts = 0
    private var escalationActive = false
    private var pendingAlertAcknowledgement = false
    private var roomTemperatureC: Float? = null
    private var customRingtoneUri: Uri? = null
    private var activeRingtone: Ringtone? = null
    private var alertDialog: AlertDialog? = null

    private val gattCallback: TimeoutGattCallback = object : TimeoutGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            super.onConnectionStateChange(gatt, status, newState)
            if (newState == BluetoothGatt.STATE_DISCONNECTED) {
                onDeviceDisconnected()
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            super.onServicesDiscovered(gatt, status)
            setNotificationForCharacteristic(
                gatt,
                GattService.AiCryService,
                GattCharacteristic.AiInferenceResult,
                Notifications.NOTIFY
            )

            // Optional temperature from Environmental Sensing.
            getCharacteristic(
                gatt,
                GattService.EnvironmentalSensing,
                GattCharacteristic.EnvironmentTemperature
            )?.let { gatt.readCharacteristic(it) }

            // Optional temperature from Health Thermometer profile.
            setNotificationForCharacteristic(
                gatt,
                GattService.HealthThermometer,
                GattCharacteristic.Temperature,
                Notifications.INDICATE
            )
        }

        override fun onDescriptorWrite(
            gatt: BluetoothGatt,
            descriptor: android.bluetooth.BluetoothGattDescriptor,
            status: Int
        ) {
            super.onDescriptorWrite(gatt, descriptor, status)
            if (descriptor.characteristic?.uuid == GattCharacteristic.AiInferenceResult.uuid) {
                getCharacteristic(
                    gatt,
                    GattService.EnvironmentalSensing,
                    GattCharacteristic.EnvironmentTemperature
                )?.let { gatt.readCharacteristic(it) }
            }
        }

        override fun onCharacteristicRead(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            status: Int
        ) {
            super.onCharacteristicRead(gatt, characteristic, status)
            if (status != BluetoothGatt.GATT_SUCCESS) {
                return
            }
            handleTemperatureCharacteristic(characteristic)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            super.onCharacteristicChanged(gatt, characteristic)
            if (characteristic.uuid == GattCharacteristic.Temperature.uuid) {
                handleTemperatureCharacteristic(characteristic)
                return
            }
            if (characteristic.uuid != GattCharacteristic.AiInferenceResult.uuid) return

            val payload = characteristic.value ?: return
            if (payload.isEmpty()) {
                return
            }

            val classId = payload[0].toInt() and 0xFF
            if (classId == TEMP_PACKET_CLASS_ID) {
                val aiPacketTempC = decodeAiPacketTemperature(payload)
                runOnUiThread {
                    if (aiPacketTempC != null) {
                        updateDisplayedRoomTemperature(aiPacketTempC)
                    }
                }
                return
            }
            if (payload.size < 2) {
                return
            }
            val rawScore = payload[1].toInt() and 0xFF
            val confidence = scoreToConfidence(rawScore)
            val mapped = mapClass(classId)
            val aiPacketTempC = decodeAiPacketTemperature(payload)
            runOnUiThread {
                updateUiFromInference(gatt, classId, rawScore, confidence, mapped, aiPacketTempC)
            }
        }
    }

    private val ringtonePickerLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val uri = result.data?.getParcelableExtra<Uri>(RingtoneManager.EXTRA_RINGTONE_PICKED_URI)
            if (uri != null) {
                customRingtoneUri = uri
                getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
                    .putString(PREF_KEY_RINGTONE_URI, uri.toString())
                    .apply()
                CustomToastManager.show(this, getString(R.string.baby_cry_ringtone_selected), 2500)
            } else {
                customRingtoneUri = null
                getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
                    .remove(PREF_KEY_RINGTONE_URI)
                    .apply()
                CustomToastManager.show(this, getString(R.string.baby_cry_ringtone_default), 2500)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityBabyCryMonitorBinding.inflate(LayoutInflater.from(this))
        setContentView(binding.root)
        AppUtil.setEdgeToEdge(window, this)
        loadPreferences()
        createNotificationChannel()
        setupToolbar()
        setupControls()
        refreshSettingLabels()
        refreshAlertUi()
        renderDisconnectedState()
    }

    override fun onResume() {
        super.onResume()
        if (serviceHasBeenSet && (service == null || !(service?.isGattConnected(connectionAddress) == true))) {
            onDeviceDisconnected()
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        val dialog = supportFragmentManager.findFragmentByTag("select_device_tag") as? SelectDeviceDialog
        dialog?.dismiss()
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return if (item.itemId == android.R.id.home) {
            gatt?.disconnect()
            onBackPressedDispatcher.onBackPressed()
            true
        } else super.onOptionsItemSelected(item)
    }

    override fun onBluetoothServiceBound() {
        serviceHasBeenSet = true
        service?.registerGattCallback(true, gattCallback)
        gatt?.discoverServices()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setHomeAsUpIndicator(R.drawable.ic_chevron_left)
            title = ""
        }
        binding.toolbar.title = ""
    }

    private fun setupControls() {
        binding.switchMonitoring.setOnCheckedChangeListener { _, isChecked ->
            powerSavingModeEnabled = isChecked
            monitoringEnabled = !powerSavingModeEnabled
            sendControlCommand(CMD_MONITORING_ENABLE, if (monitoringEnabled) 1 else 0)
            renderMonitoringModeState()
        }

        binding.switchAlerts.setOnCheckedChangeListener { _, isChecked ->
            alertsEnabled = isChecked
            sendControlCommand(CMD_ALERTS_ENABLE, if (isChecked) 1 else 0)
            if (!alertsEnabled) {
                sadDetectionsInRow = 0
                pendingAlertAcknowledgement = false
                escalationActive = false
                stopEscalationTone()
            }
            refreshAlertUi()
        }

        binding.seekThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                confidenceThreshold = progress
                refreshSettingLabels()
                sendControlCommand(CMD_THRESHOLD, confidenceThreshold)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })

        binding.seekDebounce.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                debounceCount = 1 + progress
                refreshSettingLabels()
                sendControlCommand(CMD_DEBOUNCE, debounceCount)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })

        binding.seekEscalationThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                escalationThreshold = 1 + progress
                refreshSettingLabels()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })

        binding.btnAcknowledge.setOnClickListener {
            appendHistoryEntry(getString(R.string.baby_cry_history_acknowledged))
            ignoredAlerts = 0
            escalationActive = false
            sadDetectionsInRow = 0
            pendingAlertAcknowledgement = false
            stopEscalationTone()
            refreshAlertUi()
        }

        binding.btnPickRingtone.setOnClickListener {
            val intent = Intent(RingtoneManager.ACTION_RINGTONE_PICKER).apply {
                putExtra(RingtoneManager.EXTRA_RINGTONE_TYPE, RingtoneManager.TYPE_ALARM)
                putExtra(RingtoneManager.EXTRA_RINGTONE_SHOW_DEFAULT, true)
                putExtra(
                    RingtoneManager.EXTRA_RINGTONE_EXISTING_URI,
                    customRingtoneUri ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                )
            }
            ringtonePickerLauncher.launch(intent)
        }

        binding.switchMonitoring.isChecked = powerSavingModeEnabled
        binding.switchAlerts.isChecked = alertsEnabled
        binding.seekThreshold.max = 100
        binding.seekThreshold.progress = confidenceThreshold
        binding.seekDebounce.progress = debounceCount - 1
        binding.seekEscalationThreshold.progress = escalationThreshold - 1
    }

    private fun refreshSettingLabels() {
        binding.tvThresholdValue.text = getString(R.string.baby_cry_threshold_value, confidenceThreshold)
        binding.tvDebounceValue.text = getString(R.string.baby_cry_debounce_value, debounceCount)
        binding.tvEscalationThresholdValue.text = getString(
            R.string.baby_cry_escalation_threshold_value,
            escalationThreshold
        )
    }

    private fun updateUiFromInference(
        gatt: BluetoothGatt,
        classId: Int,
        rawScore: Int,
        confidence: Float,
        mapping: ClassMapping,
        aiPacketTempC: Float?
    ) {
        val deviceName = gatt.device.name ?: gatt.device.address ?: "Device"
        binding.connectionBarText.text = getString(R.string.baby_cry_connected_to, deviceName)
        if (aiPacketTempC != null) {
            updateDisplayedRoomTemperature(aiPacketTempC)
        }
        binding.tvConfidence.text = getString(R.string.baby_cry_confidence_value, confidence)
        animateConfidenceProgress(confidence.toInt().coerceIn(0, 100))
        binding.tvCurrentClass.text = if (monitoringEnabled) {
            mapping.primary
        } else {
            getString(R.string.baby_cry_waiting_for_data)
        }
        binding.tvCrySubclass.text = getString(R.string.baby_cry_subclass_value, mapping.subclass)
        binding.tvLedState.text = getString(mapping.ledStringRes)
        applyVisualState(mapping, confidence)

        evaluateSadAlert(mapping)
        refreshAlertUi()
    }

    private fun refreshAlertUi() {
        binding.tvIgnoredAlerts.text = getString(R.string.baby_cry_ignored_alerts_value, ignoredAlerts)
        binding.tvEscalationState.text =
            if (escalationActive) getString(R.string.baby_cry_escalation_active, escalationThreshold)
            else getString(R.string.baby_cry_escalation_inactive)
        binding.tvEscalationState.setTextColor(
            ContextCompat.getColor(
                this,
                if (escalationActive) R.color.aura_red_deep else R.color.aura_text_secondary
            )
        )
    }

    private fun evaluateSadAlert(mapping: ClassMapping) {
        if (!monitoringEnabled || !alertsEnabled) {
            sadDetectionsInRow = 0
            return
        }
        sadDetectionsInRow = if (mapping.isSad) sadDetectionsInRow + 1 else 0
        if (sadDetectionsInRow >= debounceCount) {
            sadDetectionsInRow = 0
            onTriggerDetected()
        }
    }

    private fun onTriggerDetected() {
        ignoredAlerts += 1
        pendingAlertAcknowledgement = true
        appendHistoryEntry(
            getString(
                R.string.baby_cry_history_alert,
                ignoredAlerts,
                escalationThreshold
            )
        )

        showTriggerPopup(
            getString(R.string.baby_cry_trigger_popup_title),
            getString(R.string.baby_cry_trigger_popup_message)
        )
        showLocalNotification(
            getString(R.string.baby_cry_trigger_popup_title),
            getString(R.string.baby_cry_trigger_popup_message)
        )

        if (ignoredAlerts >= escalationThreshold) {
            escalationActive = true
            playEscalationTone()
            appendHistoryEntry(
                getString(R.string.baby_cry_history_escalation, escalationThreshold)
            )
            showTriggerPopup(
                getString(R.string.baby_cry_escalation_popup_title),
                getString(R.string.baby_cry_escalation_popup_message, escalationThreshold)
            )
            showLocalNotification(
                getString(R.string.baby_cry_escalation_popup_title),
                getString(R.string.baby_cry_escalation_popup_message, escalationThreshold)
            )
        }
    }

    private fun showTriggerPopup(title: String, message: String) {
        if (isFinishing || isDestroyed) return
        alertDialog?.dismiss()
        alertDialog = AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setCancelable(true)
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    private fun showLocalNotification(title: String, message: String) {
        val notification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setSmallIcon(R.drawable.redesign_ic_demo_health_thermometer)
            .setContentTitle(title)
            .setContentText(message)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(this).notify((System.currentTimeMillis() % Int.MAX_VALUE).toInt(), notification)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            ALERT_CHANNEL_ID,
            "Baby Monitor Alerts",
            NotificationManager.IMPORTANCE_HIGH
        )
        val manager = getSystemService(NotificationManager::class.java)
        manager?.createNotificationChannel(channel)
    }

    private fun playEscalationTone() {
        stopEscalationTone()
        val uri = customRingtoneUri
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
        activeRingtone = RingtoneManager.getRingtone(this, uri)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            activeRingtone?.isLooping = true
        }
        activeRingtone?.play()
    }

    private fun stopEscalationTone() {
        activeRingtone?.stop()
        activeRingtone = null
    }

    private fun loadPreferences() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        customRingtoneUri = prefs.getString(PREF_KEY_RINGTONE_URI, null)?.let(Uri::parse)
    }

    private fun sendControlCommand(command: Int, value: Int) {
        val bluetoothGatt = gatt ?: return
        val characteristic = getCharacteristic(
            bluetoothGatt,
            GattService.AiCryService,
            GattCharacteristic.AiControl
        ) ?: run {
            CustomToastManager.show(this, getString(R.string.baby_cry_control_not_supported), 1500)
            return
        }
        characteristic.value = byteArrayOf(
            command.toByte(),
            value.coerceIn(0, 255).toByte()
        )
        bluetoothGatt.writeCharacteristic(characteristic)
    }

    private fun decodeAiPacketTemperature(payload: ByteArray): Float? {
        if (payload.size < 3) return null
        val encoded = payload[2].toInt() and 0xFF
        if (encoded == AI_PACKET_TEMP_INVALID) return null
        return ((encoded - AI_PACKET_TEMP_OFFSET) * 0.5f)
    }

    private fun updateDisplayedRoomTemperature(celsius: Float) {
        roomTemperatureC = celsius
        binding.tvTemperature.text = getString(R.string.baby_cry_temperature_value, celsius)
    }

    private fun renderDisconnectedState() {
        binding.connectionBarText.text = getString(R.string.baby_cry_connected_waiting)
        binding.tvCurrentClass.text = getString(R.string.baby_cry_status_card_title_default)
        binding.tvCrySubclass.text = getString(R.string.baby_cry_subclass_default)
        binding.tvConfidence.text = getString(R.string.baby_cry_confidence_default)
        binding.tvLedState.text = getString(R.string.baby_cry_led_red)
        if (roomTemperatureC == null) {
            binding.tvTemperature.text = getString(R.string.baby_cry_temperature_na)
        }
        animateConfidenceProgress(0)
        applyVisualState(null, 0f)
    }

    private fun renderMonitoringModeState() {
        if (monitoringEnabled) {
            binding.tvCurrentClass.text = getString(R.string.baby_cry_waiting_for_data)
            binding.tvCrySubclass.text = getString(R.string.baby_cry_subclass_default)
            return
        }
        binding.tvCurrentClass.text = getString(R.string.baby_cry_power_saving_on_title)
        binding.tvCrySubclass.text = getString(R.string.baby_cry_power_saving_on_subtitle)
        binding.tvConfidence.text = getString(R.string.baby_cry_confidence_default)
        animateConfidenceProgress(0)
        applyVisualState(null, 0f)
    }

    private fun appendHistoryEntry(message: String) {
        val timestamp = DateFormat.format("dd MMM, hh:mm a", System.currentTimeMillis()).toString()
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val current = prefs.getString(PREF_KEY_HISTORY, "").orEmpty()
        val entries = mutableListOf("$timestamp  $message")
        current.lines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .take(HISTORY_LIMIT - 1)
            .forEach { entries.add(it) }
        prefs.edit().putString(PREF_KEY_HISTORY, entries.joinToString("\n")).apply()
    }

    private fun animateConfidenceProgress(targetProgress: Int) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            binding.progressConfidence.setProgress(targetProgress, true)
        } else {
            binding.progressConfidence.progress = targetProgress
        }
    }

    private fun applyVisualState(mapping: ClassMapping?, confidence: Float) {
        val isAlertState = mapping?.isSad == true && confidence >= confidenceThreshold
        val statusBackground = if (isAlertState) {
            R.drawable.aura_status_card_alert_bg
        } else {
            R.drawable.aura_status_card_bg
        }
        binding.statusCardContent.setBackgroundResource(statusBackground)

        if (mapping == null) {
            binding.statusCard.strokeColor = ContextCompat.getColor(this, R.color.aura_card_border)
            binding.tvCurrentClass.setTextColor(ContextCompat.getColor(this, R.color.aura_teal_dark))
            binding.tvStatusIcon.text = "AI"
            binding.tvStatusIcon.setTextColor(ContextCompat.getColor(this, R.color.aura_teal))
            binding.tvStatusBadgeLabel.text = getString(R.string.baby_cry_illustration_calm)
            setLedVisualMode(LedVisualMode.RED)
            return
        }

        when {
            mapping.isSad -> {
                binding.statusCard.strokeColor = ContextCompat.getColor(this, R.color.aura_led_red_border)
                binding.tvCurrentClass.setTextColor(ContextCompat.getColor(this, R.color.aura_red_deep))
                binding.tvStatusIcon.text = "CRY"
                binding.tvStatusIcon.setTextColor(ContextCompat.getColor(this, R.color.aura_red))
                binding.tvStatusBadgeLabel.text = getString(R.string.baby_cry_illustration_alert)
                setLedVisualMode(LedVisualMode.BLUE)
            }

            mapping.subclass == getString(R.string.baby_cry_class_laugh) -> {
                binding.statusCard.strokeColor = ContextCompat.getColor(this, R.color.aura_led_yellow_border)
                binding.tvCurrentClass.setTextColor(ContextCompat.getColor(this, R.color.aura_teal_dark))
                binding.tvStatusIcon.text = "JOY"
                binding.tvStatusIcon.setTextColor(ContextCompat.getColor(this, R.color.aura_orange))
                binding.tvStatusBadgeLabel.text = getString(R.string.baby_cry_illustration_laugh)
                setLedVisualMode(LedVisualMode.YELLOW)
            }

            else -> {
                binding.statusCard.strokeColor = ContextCompat.getColor(this, R.color.aura_card_border)
                binding.tvCurrentClass.setTextColor(ContextCompat.getColor(this, R.color.aura_teal_dark))
                binding.tvStatusIcon.text = "CALM"
                binding.tvStatusIcon.setTextColor(ContextCompat.getColor(this, R.color.aura_teal))
                binding.tvStatusBadgeLabel.text = getString(R.string.baby_cry_illustration_calm)
                setLedVisualMode(LedVisualMode.RED)
            }
        }
    }

    private fun setLedVisualMode(mode: LedVisualMode) {
        when (mode) {
            LedVisualMode.RED -> {
                binding.ledTile.strokeColor = ContextCompat.getColor(this, R.color.aura_led_red_border)
                binding.ledTileContent.setBackgroundResource(R.drawable.aura_led_tile_red_bg)
                binding.ledIndicator.setBackgroundResource(R.drawable.aura_led_dot_red)
            }

            LedVisualMode.YELLOW -> {
                binding.ledTile.strokeColor = ContextCompat.getColor(this, R.color.aura_led_yellow_border)
                binding.ledTileContent.setBackgroundResource(R.drawable.aura_led_tile_yellow_bg)
                binding.ledIndicator.setBackgroundResource(R.drawable.aura_led_dot_yellow)
            }

            LedVisualMode.BLUE -> {
                binding.ledTile.strokeColor = ContextCompat.getColor(this, R.color.aura_led_blue_border)
                binding.ledTileContent.setBackgroundResource(R.drawable.aura_led_tile_blue_bg)
                binding.ledIndicator.setBackgroundResource(R.drawable.aura_led_dot_blue)
            }
        }
    }

    private fun handleTemperatureCharacteristic(characteristic: BluetoothGattCharacteristic) {
        when (characteristic.uuid) {
            GattCharacteristic.EnvironmentTemperature.uuid -> {
                val raw = characteristic.getIntValue(BluetoothGattCharacteristic.FORMAT_SINT16, 0)
                if (raw != null) {
                    val celsius = convertEnvironmentalTemperature(raw)
                    runOnUiThread {
                        if (celsius != null) {
                            updateDisplayedRoomTemperature(celsius)
                        } else {
                            binding.tvTemperature.text = getString(R.string.baby_cry_temperature_na)
                        }
                    }
                }
            }

            GattCharacteristic.Temperature.uuid -> {
                val reading = TemperatureReading.fromCharacteristic(characteristic)
                val celsius = reading.getTemperature(TemperatureReading.Type.CELSIUS).toFloat()
                runOnUiThread {
                    updateDisplayedRoomTemperature(celsius)
                }
            }
        }
    }

    private fun convertEnvironmentalTemperature(raw: Int): Float? {
        // RHT service uses 0x7FFF as invalid/uninitialized temperature.
        if (raw == RHT_INVALID_RAW) return null

        // Standard Environmental Temperature scaling is usually 0.01 C.
        val cBy100 = raw / 100.0f
        if (cBy100 in VALID_TEMP_MIN_C..VALID_TEMP_MAX_C) return cBy100

        // Some firmware variants expose milli-Celsius.
        val cBy1000 = raw / 1000.0f
        if (cBy1000 in VALID_TEMP_MIN_C..VALID_TEMP_MAX_C) return cBy1000

        return null
    }

    private data class ClassMapping(
        val primary: String,
        val subclass: String,
        val isSad: Boolean,
        val ledStringRes: Int
    )

    private enum class LedVisualMode {
        RED,
        YELLOW,
        BLUE
    }

    private fun mapClass(classId: Int): ClassMapping {
        // Unified UI contract:
        // Primary state: CRY vs NOT CRY/BACKGROUND
        // Cry subtype: SAD / LAUGH / UNKNOWN
        return when (classId) {
            0 -> ClassMapping(
                primary = getString(R.string.baby_cry_primary_not_cry),
                subclass = getString(R.string.baby_cry_class_background),
                isSad = false,
                ledStringRes = R.string.baby_cry_led_red
            )

            1 -> ClassMapping(
                primary = getString(R.string.baby_cry_primary_cry),
                subclass = getString(R.string.baby_cry_class_laugh),
                isSad = false,
                ledStringRes = R.string.baby_cry_led_yellow
            )

            2, 3 -> ClassMapping(
                primary = getString(R.string.baby_cry_primary_cry),
                subclass = getString(R.string.baby_cry_class_sad),
                isSad = true,
                ledStringRes = R.string.baby_cry_led_blue
            )

            else -> ClassMapping(
                primary = getString(R.string.baby_cry_primary_not_cry),
                subclass = getString(R.string.baby_cry_class_unknown),
                isSad = false,
                ledStringRes = R.string.baby_cry_led_red
            )
        }
    }

    private fun scoreToConfidence(rawScore: Int): Float {
        val clamped = max(0, min(100, rawScore))
        return clamped.toFloat()
    }

    override fun onDestroy() {
        stopEscalationTone()
        alertDialog?.dismiss()
        alertDialog = null
        super.onDestroy()
    }

    companion object {
        private const val RHT_INVALID_RAW = 0x7FFF
        private const val VALID_TEMP_MIN_C = -40.0f
        private const val VALID_TEMP_MAX_C = 125.0f
        private const val AI_PACKET_TEMP_OFFSET = 80
        private const val AI_PACKET_TEMP_INVALID = 0xFF
        private const val TEMP_PACKET_CLASS_ID = 0xFE

        private const val ALERT_CHANNEL_ID = "baby_monitor_alerts"

        private const val PREFS_NAME = "baby_cry_monitor_prefs"
        private const val PREF_KEY_RINGTONE_URI = "ringtone_uri"
        const val PREF_KEY_HISTORY = "history_entries"
        private const val HISTORY_LIMIT = 30

        // App-level custom control protocol over AiControl characteristic.
        private const val CMD_MONITORING_ENABLE = 1
        private const val CMD_ALERTS_ENABLE = 2
        private const val CMD_THRESHOLD = 3
        private const val CMD_DEBOUNCE = 4
    }
}
