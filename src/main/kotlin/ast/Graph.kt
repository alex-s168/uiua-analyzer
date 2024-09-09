package me.alex_s168.uiua.ast

import blitz.*
import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.PushFnInstr
import me.alex_s168.uiua.PushFnRefInstr
import kotlin.math.max

fun List<ASTRoot>.genGraph(): String {
    val out = mutableListOf<String>()

    out += "digraph g {"

    val flattened = map { it.children.flatMap { it.flatten() } }

    var nextId = 0
    val node2id = flattened.flatten()
        .associateWith { "node${nextId ++}" }

    flattened.forEachIndexed { idx, it ->
        val root = get(idx)
        out += " subgraph \"cluster_${root.functionName!!}\" {"
        out += "  style=filled;"
        out += "  color=lightgrey;"

        it.forEach { v ->
            val k = node2id[v]!!

            if (v.value.isA && v.value.getA().instr.let { it is PushFnInstr || it is PushFnRefInstr })
                return@forEach

            val label = v.value
                .mapA {
                    when (it.instr) {
                        is PrimitiveInstr -> "${it.instr.id}${ it.instr.param?.let { ":$it" } ?: "" }"

                        is NumImmInstr -> it.instr.value.toString()

                        else -> it.instr.toString()
                    }
                }
                .mapBA { "arg(${it.id})" }
                .mapBB { "..." }
                .mapB { it.flatten() }
                .flatten()

            val numExtraArgs = v.value.getAOrNull()?.children?.size?.let { it1 -> max(0, it1 - 1) } ?: 0
            val extraArgs = List(numExtraArgs) { "<f${it + 1}> x" }
                .joinToString("| ")
                .let { if (it.isNotEmpty()) "| $it" else "" }

            out += "  \"$k\" ["
            out += "   label = \"<f0> $label$extraArgs\""
            out += "   shape = \"record\""
            out += "  ];"
        }

        out += " }"
    }

    fun parent(from: AstNode) =
        node2id.keys.find { it.value.isA && it.value.isA && from in it.value.getA().children }

    node2id.forEach { (from, fromKey) ->
        fun outputTo(to: AstNode) {
            if (to.value.getAOrNull()?.instr?.let { it is PushFnInstr || it is PushFnRefInstr } == true) {
                val a = to.value.getA().instr

                val fnName =  if (a is PushFnInstr) {
                    a.fn.value.getA()
                } else if (a is PushFnRefInstr) {
                    a.fn
                } else null

                fnName?.let { name ->
                    val calling = find { it.functionName == name }!!
                    val first = node2id[calling.children.first()]!!
                    out += " $first:f0 -> \"$fromKey\":f0 [ltail = \"cluster_$name\"]"
                }

                return
            }

            val toKey = node2id[to]!!
            val argIdx = parent(to)?.value?.getA()?.children?.indexOf(to) ?: 0
            out += " \"$toKey\":f0 -> \"$fromKey\":f$argIdx"
        }

        if (from.value.isA) {
            val a = from.value.getA()

            a.children.forEach { outputTo(it) }
        }
        else {
            from.value.getBBOrNull()?.of?.let { outputTo(it) }
        }
    }

    out += "}"

    return out.joinToString("\n")
}