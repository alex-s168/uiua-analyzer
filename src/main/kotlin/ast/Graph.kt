package me.alex_s168.uiua.ast

import blitz.*
import blitz.collections.contents
import me.alex_s168.uiua.*
import kotlin.math.max
import kotlin.math.min

fun List<ASTRoot>.genGraph(): String {
    val out = mutableListOf<String>()

    out += "digraph g {"

    val flattened = map { it.children.flatMap { it.flatten() } }
    val flatFlattened = flattened.flatten().toSet()

    var nextId = 0
    val node2id = flattened.flatten()
        .associateWith { "node${nextId ++}" }

    fun parent(from: AstNode) =
        node2id.keys.find { it.value.isA && it.value.isA && from in it.value.getA().children }

    fun parentChildren(from: AstNode) =
        parent(from)?.value?.getA()?.children
            ?: find { from in it.children }?.children

    fun trunc(it: String) =
        if (it.length > 5) it.take(3) + "..."
        else it

    flattened.forEachIndexed { idx, it ->
        val root = get(idx)
        out += " subgraph \"cluster_${root.functionName!!}\" {"
        out += "  style=filled;"
        out += "  color=lightgrey;"

        it.forEach { v ->
            val k = node2id[v]!!

            if (v.value.isA && v.value.getA().instr.let { it is PushFnInstr || it is PushFnRefInstr })
                return@forEach

            if (v.value.isB && v.value.getB().isB)
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

            val childrenSize = v.value.getAOrNull()?.children?.size
            val numExtraArgs = childrenSize?.let { it1 -> max(0, it1 - 1) } ?: 0
            val numExtraRets = flatFlattened.count { it.value.getBBOrNull()?.let { it.of == v } ?: false } ?: 0
            if (numExtraRets > 0) {
                println(v.value.getA().instr.toString())
            }
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
                    val argIdx = parentChildren(to)?.indexOf(to) ?: 0
                    out += " $first:f0 -> \"$fromKey\":$argIdx [ltail = \"cluster_$name\"]"
                }

                return
            }

            if (to.value.getBBOrNull() != null) {
                val extend = to.value.getBBOrNull()!!.of
                val fromParent = parent(to)!!
                val parentKey = node2id[fromParent]!!
                val extendKey = node2id[extend]!!
                val argIdx = parentChildren(to)!!.indexOf(to)
                out += " \"$extendKey\":f1 -> \"$parentKey\":f$argIdx"

                return
            }

            val argIdx = parentChildren(to)?.indexOf(to) ?: 0
            val toKey = node2id[to]!!
            out += " \"$toKey\":f0 -> \"$fromKey\":f$argIdx"
        }

        if (from.value.isA) {
            val a = from.value.getA()

            a.children.forEach { outputTo(it) }
        }
    }

    out += "}"

    return out.joinToString("\n")
}