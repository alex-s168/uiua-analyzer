package me.alex_s168.uiua

import blitz.Either
import blitz.collections.*
import blitz.io.Path
import blitz.io.read
import blitz.parse.JSON
import blitz.str.flattenToString
import kotlin.streams.toList

data class Signature(
    val inputs: Int,
    val outputs: Int
) {
    companion object {
        fun parse(arr: List<JSON.Element>): Signature =
            Signature(
                arr[0].num.toInt(),
                arr[1].num.toInt()
            )
    }
}

data class SpanRef(
    val sourceFile: String,
    val start: Loc,
    val end: Loc,
) {
    companion object {
        fun parse(arr: List<JSON.Element>): SpanRef =
            SpanRef(
                arr[0].str,
                Loc.parse(arr[1].arr),
                Loc.parse(arr[2].arr)
            )
    }
}

data class Loc(
    val line: Int,
    val col: Int,
    val bytePos: Int,
    val charPos: Int
) {
    companion object {
        fun parse(arr: List<JSON.Element>): Loc =
            Loc(
                arr[0].num.toInt(),
                arr[1].num.toInt(),
                arr[2].num.toInt(),
                arr[3].num.toInt()
            )
    }
}

data class InstSpan(
    val start: Int,
    val len: Int,
) {
    companion object {
        fun parse(arr: List<JSON.Element>): InstSpan =
            InstSpan(
                arr[0].num.toInt(),
                arr[1].num.toInt()
            )
    }
}

data class Function(
    val value: Either<String, SpanRef>,
    var children: List<Instr>,
    val signature: Signature,
    val loc: InstSpan,
    val rec: Boolean,
)

abstract class Instr

data class PrimitiveInstr(
    val id: String,
    val loc: Int
): Instr()

abstract class ImmInstr: Instr()

data class ArrImmInstr(
    val type: ArrayType,
    val values: Either<List<Int>, List<Double>>
): ImmInstr()

data class NumImmInstr(
    val value: Double,
): ImmInstr()

data class PushFnInstr(
    val fn: Function
): ImmInstr()

data class CommentInstr(
    val comment: String
): Instr()

data class Assembly(
    val instructions: MutableList<Instr>,
    val sourceFiles: Map<String, String>,
) {
    companion object {
        fun parse(text: String): Assembly {
            val sections = text
                .trimEnd()
                .split("\n\n")
                .asSequence()
                .mapIndexed { index, s ->
                    if (index == 0) {
                        "INSTRUCTIONS" to s.split('\n')
                    }
                    else {
                        val spl = s.split('\n')
                        spl.first() to spl.drop(1)
                    }
                }
                .toMap()

            val instrs = mutableListOf<Instr>()
            sections["INSTRUCTIONS"]!!
                .forEach { instr ->
                    val parsed = if (instr.startsWith('[')) {
                        val all = JSON.parse(instr)!!.arr
                        val shape = all[0].arr
                        val data = all[1]

                        val elemType = if (data.isStr()) Types.byte else Types.double

                        if (shape.isEmpty()) {
                            NumImmInstr(
                                data.arr[0].num
                            )
                        } else {
                            val type = shape.map { it.num.toInt() }.shapeToType(elemType)
                            ArrImmInstr(
                                type,
                                if (data.isStr()) Either.ofA(data.str.chars().toList())
                                else Either.ofB(data.arr.map { it.num })
                            )
                        }
                    }
                    else if (instr.startsWith("comment")) {
                        CommentInstr(instr.substringAfter("comment").trim())
                    }
                    else if (instr.startsWith("push_func")) {
                        val arr = JSON.parse(instr.substringAfter("push_func").trim())!!.arr
                        val span = SpanRef.parse(arr[0].arr)
                        val signature = Signature.parse(arr[1].arr)
                        val loc = InstSpan.parse(arr[2].arr)
                        PushFnInstr(Function(
                            Either.ofB(span),
                            listOf(),
                            signature,
                            loc,
                            arr[4].bool
                        ))
                    }
                    else {
                        kotlin.runCatching {
                            val (id, loc) = instr.split(' ')
                            PrimitiveInstr(id, loc.toInt())
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

            val files = sections["FILES"]!!
                .associate {
                    val (name, content) = it.split(' ', limit = 2)
                    name to unescape(content)
                }

            return Assembly(instrs, files)
        }
    }
}

fun main() {
    val test = Path.of("/home/alex/uiua-analyzer/test.uasm")
        .getFile()
        .read()
        .stringify()
        .flattenToString()
    val assembly = Assembly.parse(test)
    assembly.instructions.removeAll { it is CommentInstr }
    println(assembly.instructions.joinToString("\n"))
}