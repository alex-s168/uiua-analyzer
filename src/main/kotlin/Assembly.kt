package me.alex_s168.uiua

import blitz.Either
import blitz.collections.mergeNeighbors
import blitz.collections.nullIfEmpty
import blitz.collections.substringAfter
import blitz.parse.JSON
import blitz.parse.JSON.asArr
import blitz.parse.JSON.asNum
import blitz.parse.JSON.asStr
import blitz.parse.comb2.unwrap
import blitz.startsWithCase
import blitz.str.unescape
import blitz.str.splitWithNesting
import blitz.switch
import blitz.flatMap
import blitz.collections.RefVec

private fun parseInstrV2(input: JSON.Element): List<Instr> =
    when (input.kind) {
        JSON.Element.NUM -> listOf(NumImmInstr(input.asNum()))

        JSON.Element.STR -> listOf(PrimitiveInstr(Prims.all2[input.asStr()]!!))

        JSON.Element.ARR -> {
            val args = input.asArr()
            if (args[0].kind != JSON.Element.STR) {
                listOf(ArrImmInstr(
                    Types.array(Types.tbd, args.size),
                    Either.ofB(args.mapTo(mutableListOf()) { it.asNum() })
                ))
            } else {
                val primStr = args[0].asStr()
                val prim = Prims.all2[primStr] 
                    ?: error("primitive $primStr not found")

                if (args[1].kind == JSON.Element.NUM) {
                    listOf(
                        PrimitiveInstr(prim, SpanRef(listOf(args[1].asNum().toInt())))
                    )
                } else {
                    val innerFn = args[1].asArr()[0].asArr()
                    var innerFnInstrs = innerFn[2].asArr()
                    // TODO: annoy kai about this and get him to change it
                    if (innerFnInstrs.size == 2 && innerFnInstrs[0].kind == JSON.Element.STR && innerFnInstrs[1].kind == JSON.Element.NUM)
                        innerFnInstrs = RefVec.of(JSON.Element.newArr(innerFnInstrs))

                    listOf(
                        PushFnInstr(Function(
                            Either.ofA(anonFnName()),
                            innerFnInstrs.flatMapTo(mutableListOf(), ::parseInstrV2),
                            Signature(innerFn[0].asNum().toInt(), innerFn[1].asNum().toInt()),
                            null, false
                        )),
                        PrimitiveInstr(prim)
                    )
                }
            }
        }

        else -> error("error in instr: $input")
    }

private fun parseInstr(instrIn: String): Instr? {
    val instr = instrIn.trim()
    if (instr.isEmpty()) return null

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
                    PrimitiveInstr(Prims.Front.REDUCE_DEPTH, SpanRef(listOf(loc.toInt())), depth)
                }

                "TRANSPOSE_N" -> {
                    val to = when (value.toInt()) {
                        -1 -> Prims.Front.UN_TRANSPOSE
                        1 -> Prims.TRANSPOSE
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
            val arr = JSON.parse(instr.substringAfter(it)).unwrap().asArr()
            PushFnInstr.parse(arr)
        },
        Regex("(?i)copy_?to_?temp *\\[(.*?)\\]") startsWithCase {
            val (stack, count) = it.groupValues[1].split(',')
            CopyTempStackInstr(stack, count.toInt())
        },
        Regex("(?i)push_?temp *\\[(.*?)\\]") startsWithCase {
            val (stack, count) = it.groupValues[1].split(',')
            PushTempStackInstr(stack, count.toInt())
        },
        Regex("(?i)pop_?temp *\\[(.*?)\\]") startsWithCase {
            val (stack, count) = it.groupValues[1]
                .split(',')
            PopTempStackInstr(stack, count.toInt())
        },
        Regex("\"(.*)\"") startsWithCase {
            FlagInstr(it.groupValues[1])
        },
        Regex("(?i)comment *(.*?)") startsWithCase {
            CommentInstr(it.groupValues[1])
        }
    ) { s ->
        kotlin.runCatching {
            val (id, loc) = s.split(' ')
            val prim = Prims.all2[id.uppercase()] ?: error("unknown primitive $id")
            PrimitiveInstr(prim, SpanRef(listOf(loc.toInt())))
        }.getOrElse {
            NumImmInstr(s.toDouble())
        }
    }

    return parsed
}

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
                    instrs += parseInstr(instrIn)
                        ?: return@forEach
                }

            val asmFns = mutableListOf<List<Instr>>()
            sections["FUNCTIONS"]
                ?.forEach { instrs ->
                    asmFns += JSON.parse(instrs)
                        .unwrap()
                        .asArr()
                        .flatMapTo(mutableListOf(), ::parseInstrV2)
                }

            fun InstSpan.get(): List<Instr> =
                impl.flatMap(
                    { instrs.subList(it.start, it.start + it.len).toList() },
                    { asmFns[it.loc] }
                )

            for (instr in instrs) {
                if (instr is PushFnInstr)
                    instr.fn.children = instr.fn.loc!!.get()
            }

            val spans = sections["SPANS"]!!
                .map(Span::parseNew)

            val functions = sections["BINDINGS"]!!
                .asSequence()
                .filter { it.startsWith("func ") }
                .map {
                    val j = JSON.parse(it.substringAfter("func ")).assertA().asArr()
                    val fn = PushFnInstr.parse(j).fn
                    fn.children = fn.loc!!.get()
                    fn
                }
                .associateByTo(mutableMapOf()) {
                    it.value.assertA()
                }

            val entryFn = sections["TOP SLICES"]
                ?.flatMap {
                    val (a, b) = it.split(' ').map(String::toInt)
                    instrs.subList(a, a + b)
                } ?: listOf()

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
