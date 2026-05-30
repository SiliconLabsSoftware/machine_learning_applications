package com.siliconlabs.bledemo.home_screen.fragments

import android.content.Context
import android.graphics.Typeface
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import com.siliconlabs.bledemo.R
import com.siliconlabs.bledemo.databinding.FragmentHistoryBinding
import com.siliconlabs.bledemo.features.demo.babycry.BabyCryMonitorActivity
import com.google.android.material.card.MaterialCardView

class HistoryFragment : Fragment() {

    private lateinit var binding: FragmentHistoryBinding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = FragmentHistoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        activity?.title = getString(R.string.main_navigation_history_title)
    }

    override fun onResume() {
        super.onResume()
        val history = requireContext()
            .getSharedPreferences("baby_cry_monitor_prefs", Context.MODE_PRIVATE)
            .getString(BabyCryMonitorActivity.PREF_KEY_HISTORY, "")
            .orEmpty()
            .trim()

        binding.tvHistoryEmpty.isVisible = history.isEmpty()
        binding.historyList.isVisible = history.isNotEmpty()
        renderHistoryCards(history)
    }

    private fun renderHistoryCards(history: String) {
        binding.historyList.removeAllViews()
        if (history.isEmpty()) return

        val entries = history.lines().map { it.trim() }.filter { it.isNotEmpty() }
        entries.forEachIndexed { index, entry ->
            binding.historyList.addView(createHistoryCard(entry, index == 0))
        }
    }

    private fun createHistoryCard(entry: String, highlight: Boolean): View {
        val context = requireContext()
        val outerMargin = dp(6)
        val card = MaterialCardView(context).apply {
            radius = dp(18).toFloat()
            cardElevation = dp(2).toFloat()
            strokeWidth = dp(1)
            setCardBackgroundColor(resources.getColor(R.color.aura_card, context.theme))
            strokeColor = resources.getColor(
                if (highlight) R.color.aura_navy else R.color.aura_card_border,
                context.theme
            )
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = if (binding.historyList.childCount == 0) 0 else outerMargin
            }
        }

        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(14))
        }

        val splitIndex = entry.indexOf("  ")
        val timestamp = if (splitIndex >= 0) entry.substring(0, splitIndex).trim() else ""
        val message = if (splitIndex >= 0) entry.substring(splitIndex).trim() else entry

        if (timestamp.isNotEmpty()) {
            container.addView(TextView(context).apply {
                text = timestamp
                setTextColor(resources.getColor(R.color.aura_text_secondary, context.theme))
                textSize = 12f
                setTypeface(typeface, Typeface.BOLD)
            })
        }

        container.addView(TextView(context).apply {
            text = message
            setTextColor(resources.getColor(R.color.aura_text_primary, context.theme))
            textSize = 14f
            if (timestamp.isNotEmpty()) {
                setPadding(0, dp(6), 0, 0)
            }
        })

        card.addView(container)
        return card
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }
}
