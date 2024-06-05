package me.alex_s168.uiua.mlir

fun String.legalizeMLIR(): String =
    replace("[", "_start_").replace("]", "_end_").replace("?", "_maybe_")