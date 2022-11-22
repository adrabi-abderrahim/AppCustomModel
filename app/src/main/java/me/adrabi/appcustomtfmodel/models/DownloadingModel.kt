package me.adrabi.appcustomtfmodel.models


data class DownloadingModel(
    val status: Int,
    val total: Long,
    val downloaded: Long
)