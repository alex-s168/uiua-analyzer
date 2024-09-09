package me.alex_s168.uiua.ast

import blitz.flatten
import blitz.getBBOrNull
import blitz.mapBA
import blitz.mapBB
import me.alex_s168.uiua.PushFnInstr
import me.alex_s168.uiua.PushFnRefInstr

fun List<ASTRoot>.genGraph(): String {
    val out = mutableListOf<String>()

    out += "digraph g {"

    val flattened = map { it.children.flatMap { it.flatten() } }

    var nextId = 0
    val node2id = flattened.flatten()
        .associateWith { "node${nextId ++}" }

    flattened.forEach {
        out += " subgraph node${nextId++} {"
        out += "  style=filled;"
        out += "  color=lightgrey;"

        it.forEach { v ->
            val k = node2id[v]!!

            val label = v.value
                .mapA {
                    when (it.instr) {
                        is PushFnInstr,
                        is PushFnRefInstr -> "fnref"

                        else -> it.instr.toString()
                    }
                }
                .mapBA { "arg(${it.id})" }
                .mapBB { "..." }
                .mapB { it.flatten() }
                .flatten()

            out += "  \"$k\" ["
            out += "   label = \"$label\""
            out += "   shape = \"record\""
            out += "  ];"
        }

        out += " }"
    }

    node2id.forEach { (from, fromKey) ->
        if (from.value.isA) {
            val a = from.value.getA()

            a.children.forEach { to ->
                val toKey = node2id[to]!!
                out += " \"$toKey\" -> \"$fromKey\""
            }
        }
        else {
            from.value.getBBOrNull()?.of?.let { to ->
                val toKey = node2id[to]!!
                out += " \"$toKey\" -> \"$fromKey\""
            }
        }
    }

    out += "}"

    return out.joinToString("\n")
}