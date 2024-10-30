package me.alex_s168.uiua

import blitz.Either
import blitz.collections.mergeNeighbors
import blitz.collections.nullIfEmpty
import blitz.collections.substringAfter
import blitz.parse.JSON
import blitz.parse.JSON.asArr
import blitz.parse.comb2.unwrap
import blitz.startsWithCase
import blitz.str.unescape
import blitz.switch

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

                    val parsed = instr.switch(
                        Regex.fromLiteral("[{") startsWithCase {
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

                                "TRANSPOSE_N" -> {
                                    val to = when (value.toInt()) {
                                        -1 -> Prim.Front.UN_TRANSPOSE
                                        1 -> Prim.TRANSPOSE
                                        else -> error("unsupported transpose_n amount")
                                    }
                                    PrimitiveInstr(to, SpanRef(listOf(loc.toInt())))
                                }

                                else -> error("unsupported complicated instruction $kind")
                            }
                        },
                        Regex.fromLiteral("[") startsWithCase {
                            val all = JSON.parse(instr).assertA().asArr()

                            var rank = 1
                            var iter = all
                            while (iter.firstOrNull()?.kind == JSON.Element.ARR) {
                                iter = iter.first().asArr()
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
                        },
                        Regex.fromLiteral("# ") startsWithCase {
                            CommentInstr(instr.substringAfter(it).trim())
                        },
                        Regex("(?i)push_?func +") startsWithCase {
                            val arr = JSON.parse(instr.substringAfter(it).also(::println)).unwrap().asArr()
                            PushFnInstr.parse(arr)
                        },
                        Regex("(?i)copy_?to_?temp *\\[(.*)\\]") startsWithCase {
                            val (stack, count) = it.groupValues[1].split(',')
                            CopyTempStackInstr(stack, count.toInt())
                        },
                        Regex("(?i)push_?temp *\\[(.*)\\]") startsWithCase {
                            val (stack, count) = it.groupValues[1].split(',')
                            PushTempStackInstr(stack, count.toInt())
                        },
                        Regex("(?i)pop_?temp *\\[(.*)\\]") startsWithCase {
                            val (stack, count) = it.groupValues[1]
                                .split(',')
                            PopTempStackInstr(stack, count.toInt())
                        },
                        Regex("\"(.*)\"") startsWithCase {
                            FlagInstr(it.groupValues[1])
                        },
                        Regex("(?i)comment *(.*)") startsWithCase {
                            CommentInstr(it.groupValues[1])
                        }
                    ) { s ->
                        kotlin.runCatching {
                            val (id, loc) = s.split(' ')
                            PrimitiveInstr(id.uppercase(), SpanRef(listOf(loc.toInt())))
                        }.getOrElse {
                            NumImmInstr(s.toDouble())
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
                    val j = JSON.parse(it.substringAfter("func ")).assertA().asArr()
                    val fn = PushFnInstr.parse(j).fn
                    fn.children = instrs.subList(fn.loc!!.start, fn.loc.start + fn.loc.len).toList()
                    fn
                }
                .associateByTo(mutableMapOf()) {
                    it.value.assertA()
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