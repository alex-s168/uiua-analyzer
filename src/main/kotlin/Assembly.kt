package me.alex_s168.uiua

import blitz.Either
import blitz.collections.mergeNeighbors
import blitz.collections.nullIfEmpty
import blitz.parse.JSON
import blitz.str.unescape
import kotlin.streams.toList

data class Assembly(
    val instructions: MutableList<Instr>,
    val sourceFiles: Map<String, String>,
    val functions: Map<String, Function>,
    val spans: List<Span>,
) {
    companion object {
        fun parse(text: String): Assembly {
            val sections = text
                .trimEnd()
                .lines()
                .mergeNeighbors { it.isEmpty() }
                .mapNotNull { it.second.nullIfEmpty() }
                .asSequence()
                .mapIndexed { index, s ->
                    if (index == 0) {
                        "INSTRUCTIONS" to s
                    }
                    else {
                        s.first() to s.drop(1)
                    }
                }
                .toMap()

            val instrs = mutableListOf<Instr>()
            sections["INSTRUCTIONS"]!!
                .forEach { instrIn ->
                    val instr = instrIn.trim()
                    if (instr.isEmpty()) return@forEach

                    val parsed = if (instr.startsWith("[{")) {
                        val (l, loc) = instr
                            .drop(1)
                            .dropLast(1)
                            .split(',')

                        val (kind, value) = l.drop(1).dropLast(1).split(':')

                        when (kind.drop(1).dropLast(1)) {
                            "REDUCE_DEPTH" -> {
                                val depth = value.toInt()
                                PrimitiveInstr(Prim.Front.REDUCE_DEPTH, SpanRef(listOf(loc.toInt())), depth)
                            }

                            else -> error("unsupported $kind")
                        }
                    } else if (instr.startsWith('[')) {
                        val all = JSON.parse(instr)!!.arr
                        val shape = all[0].arr
                        val data = all[1]

                        val elemType = if (data.isStr()) Types.byte else Types.double

                        if (shape.isEmpty()) {
                            NumImmInstr(
                                data.arr[0].num
                            )
                        } else {
                            val type = shape.map { it.num.toInt() }.shapeToArrType(elemType)
                            ArrImmInstr(
                                type,
                                if (data.isStr()) Either.ofA(data.str.chars().toList())
                                else Either.ofB(data.arr.map { it.num })
                            )
                        }
                    }
                    else if (instr.startsWith("# ")) {
                        CommentInstr(instr.substringAfter("# ").trim())
                    }
                    else if (instr.startsWith("push_func ")) {
                        val arr = JSON.parse(instr.substringAfter("push_func ").trim())!!.arr
                        PushFnInstr.parse(arr)
                    }
                    else if (instr.startsWith("copy_to_temp ")) {
                        val (stack, count) = instr
                            .substringAfter("copy_to_temp [")
                            .substringBeforeLast(']')
                            .split(',')
                        CopyTempStackInstr(stack, count.toInt())
                    }
                    else if (instr.startsWith("pop_temp ")) {
                        val (stack, count) = instr
                            .substringAfter("pop_temp [")
                            .substringBeforeLast(']')
                            .split(',')
                        PopTempStackInstr(stack, count.toInt())
                    }
                    else if (instr.startsWith('"')) {
                        FlagInstr(instr.drop(1).dropLast(1))
                    }
                    else {
                        kotlin.runCatching {
                            val (id, loc) = instr.split(' ')
                            PrimitiveInstr(id, SpanRef(listOf(loc.toInt())))
                        }.getOrElse {
                            NumImmInstr(instr.toDouble())
                        }
                    }

                    instrs.add(parsed)
                }

            for (instr in instrs) {
                if (instr is PushFnInstr)
                    instr.fn.children = instrs.subList(instr.fn.loc.start, instr.fn.loc.start + instr.fn.loc.len).toList()
            }

            val slices = sections["TOP SLICES"]!!
                .map {
                    val (a, b) = it.split(' ')
                    a to b
                }

            val spans = sections["SPANS"]!!
                .map(Span::parseNew)

            val functions = sections["BINDINGS"]!!
                .asSequence()
                .filter { it.startsWith("func ") }
                .map {
                    val j = JSON.parse(it.substringAfter("func "))!!.arr
                    val fn = PushFnInstr.parse(j).fn
                    fn.children = instrs.subList(fn.loc.start, fn.loc.start + fn.loc.len).toList()
                    fn
                }
                .associateBy {
                    it.value.getA()
                }

            val files = sections["FILES"]!!
                .associate {
                    val (name, content) = it.split(' ', limit = 2)
                    name to unescape(content)
                }

            return Assembly(instrs, files, functions, spans)
        }
    }
}