package com.siliconlabs.bledemo.home_screen.fragments

import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.siliconlabs.bledemo.BuildConfig
import com.siliconlabs.bledemo.R
import com.siliconlabs.bledemo.databinding.FragmentSettingsBinding
import com.siliconlabs.bledemo.home_screen.dialogs.DeviceInformationDialog

class SettingsFragment : Fragment() {

    private lateinit var _binding: FragmentSettingsBinding

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(inflater)
        return _binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        activity?.title = getString(R.string.action_settings)
        bindDeviceSummary()
        setupUiListeners()
    }

    private fun bindDeviceSummary() {
        _binding.apply {
            tvDeviceName.text = getString(R.string.settings_device_name_label, Build.MODEL)
            tvDeviceModel.text = getString(
                R.string.settings_model_label,
                "${Build.MANUFACTURER} ${Build.PRODUCT}"
            )
            tvAndroidVersion.text =
                getString(R.string.settings_android_label, Build.VERSION.RELEASE ?: "Unknown")
            tvBuildVersion.text = getString(R.string.settings_build_label, Build.ID)
            dialogHelpVersionText.text = getString(R.string.version_text, BuildConfig.VERSION_NAME)
        }
    }

    private fun setupUiListeners() {
        _binding.apply {
            btnDeviceInfo.setOnClickListener {
                DeviceInformationDialog().show(childFragmentManager, "Device_Information_Dialog")
            }
        }
    }
}
