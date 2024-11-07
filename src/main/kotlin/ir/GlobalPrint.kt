package me.alex_s168.uiua.ir

import java.io.PrintWriter

val globalPrint = GlobalPass("print all") { blocks, run: ((PrintWriter) -> Unit) -> Unit ->
    run { dest ->
        blocks.values.forEach {
            dest.println(it)
            dest.println()
        }
    }
}