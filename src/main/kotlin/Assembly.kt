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

                        var rank = 1
                        var iter = all
                        while (iter.firstOrNull()?.isArr() == true) {
                            iter = iter.firstOrNull()!!.arr
                            rank ++
                        }

                        val data = instr.drop(1).dropLast(1)
                            .replace(",[", "")
                            .replace("[", "")
                            .replace("]", "")
                            .split(',')
                            .map { it.toDouble() }

                        val elemType = Types.double
                        val type = Types.ndarray(rank, elemType)
                        ArrImmInstr(
                            type,
                            Either.ofB(data)
                        )
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
                    else if (instr.startsWith("push_temp ")) {
                        val (stack, count) = instr
                            .substringAfter("push_temp [")
                            .substringBeforeLast(']')
                            .split(',')
                        PushTempStackInstr(stack, count.toInt())
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
                    instr.fn.children = instrs.subList(instr.fn.loc!!.start, instr.fn.loc.start + instr.fn.loc.len).toList()
            }

            val spans = sections["SPANS"]!!
                .map(Span::parseNew)

            val functions = sections["BINDINGS"]!!
                .asSequence()
                .filter { it.startsWith("func ") }
                .map {
                    val j = JSON.parse(it.substringAfter("func "))!!.arr
                    val fn = PushFnInstr.parse(j).fn
                    fn.children = instrs.subList(fn.loc!!.start, fn.loc.start + fn.loc.len).toList()
                    fn
                }
                .associateByTo(mutableMapOf()) {
                    it.value.getA()
                }

            val entryFn = sections["TOP SLICES"]!!
                .flatMap {
                    val (a, b) = it.split(' ').map(String::toInt)
                    instrs.subList(a, a + b)
                }

            if (entryFn.isNotEmpty()) {
                functions["_\$main"] = Function(
                    value = Either.ofA(""),
                    children = entryFn,
                    signature = null,
                    loc = null,
                    rec = false,
                )
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