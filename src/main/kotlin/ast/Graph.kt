package me.alex_s168.uiua.ast

import blitz.*
import blitz.collections.contents
import me.alex_s168.uiua.*
import kotlin.math.max

fun List<ASTRoot>.genGraph(): String {
    val out = mutableListOf<String>()

    out += "digraph g {"

    val flattened = map { it.children.flatMap { it.flatten() } }
    val flatFlattened = flattened.flatten().toSet()

    var nextId = 0
    val node2id = flattened.flatten()
        .associateWith { "node${nextId ++}" }

    fun parent(from: AstNode) =
        node2id.keys.find { it.value.isA() && from in it.value.a!!.children }

    fun parentChildren(from: AstNode) =
        parent(from)?.value?.a?.children
            ?: find { from in it.children }?.children

    fun trunc(it: String) =
        if (it.length > 5) it.take(3) + "..."
        else it

    flattened.forEachIndexed { idx, it ->
        val root = get(idx)
        out += " subgraph \"cluster_${root.functionName!!}\" {"
        out += "  label=\"${root.functionName!!}\";"
        out += "  style=filled;"
        out += "  color=lightgrey;"

        it.forEach { v ->
            val k = node2id[v]!!

            if (v.value.isA() && v.value.a!!.instr.let { it is PushFnInstr || it is PushFnRefInstr })
                return@forEach

            if (v.value.isB() && v.value.b!!.isB())
                return@forEach

            val label = v.value
                .mapA {
                    when (it.instr) {
                        is PrimitiveInstr -> "${it.instr.id}${ it.instr.param?.let { ":$it" } ?: "" }"

                        is NumImmInstr -> it.instr.value.toString().let(::trunc)

                        is ArrImmInstr -> it.instr.values.flatten()
                            .map { trunc(it.toString()) }
                            .contents.toString()

                        else -> it.instr.toString()
                    }
                }
                .mapBA { "arg(${it.id})" }
                .mapB { it.flatten() }
                .flatten()

            val childrenSize = v.value.a?.children?.size
            val numExtraArgs = childrenSize?.let { it1 -> max(0, it1 - 1) } ?: 0
            val numExtraRets = flatFlattened.count { it.value.getBBOrNull()?.let { it.of == v } ?: false }
            val rem = max(0, numExtraRets - (childrenSize?.let { max(0,it -1) } ?: 0))
            val extraArgs = (List(numExtraArgs) { "<f${it + 1}> x" } + List(rem) { "<f${it + numExtraArgs + 1}> ." })
                .joinToString("| ")
                .let { if (it.isNotEmpty()) "| $it" else "" }

            out += "  \"$k\" ["
            out += "   label = \"<f0> $label$extraArgs\""
            out += "   shape = \"record\""
            out += "  ];"
        }

        out += " }"
    }

    node2id.forEach { (to, toKey) ->
        fun inputFrom(argIdx: Int, from: AstNode) {
            if (from.value.a?.instr?.let { it is PushFnInstr || it is PushFnRefInstr } == true) {
                val a = from.value.a!!.instr

                val fnName =  if (a is PushFnInstr) {
                    a.fn.value.a!!
                } else if (a is PushFnRefInstr) {
                    a.fn
                } else null

                fnName?.let { name ->
                    val calling = find { it.functionName == name }!!
                    val first = node2id[calling.children.first()]!!
                    out += " $first:f0 -> \"$toKey\":f$argIdx [ltail = \"cluster_$name\"; color = \"blue\"]"
                }

                return
            }

            if (from.value.getBBOrNull() != null) {
                val (extend, outIdx) = from.value.getBBOrNull()!!.let { it.of to it.resIdx }
                val extendKey = node2id[extend]!!
                out += " \"$extendKey\":f${outIdx} -> \"$toKey\":f$argIdx"

                return
            }

            val color = from.value
                .mapA { "black" }
                .mapBA { "red" }
                .mapBB { "" }
                .mapB { it.flatten() }
                .flatten()

            val fromKey = node2id[from]!!
            out += " \"$fromKey\":f0 -> \"$toKey\":f$argIdx [color = \"$color\"]"
        }

        if (to.value.isA()) {
            val a = to.value.a!!

            a.children.forEachIndexed { i, it -> inputFrom(i, it) }
        }
    }

    out += "}"

    return out.joinToString("\n")
}