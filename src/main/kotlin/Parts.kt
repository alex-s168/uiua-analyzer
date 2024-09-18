package me.alex_s168.uiua

import blitz.Either
import blitz.parse.JSON

data class Signature(
    val inputs: Int,
    val outputs: Int
) {
    fun mapIns(fn: (Int) -> Int) =
        Signature(fn(inputs), outputs)

    fun mapOuts(fn: (Int) -> Int) =
        Signature(inputs, fn(outputs))

    companion object {
        fun parse(arr: List<JSON.Element>): Signature =
            Signature(
                arr[0].num.toInt(),
                arr[1].num.toInt()
            )
    }
}

data class Span(
    val sourceFile: String,
    val start: Loc,
    val end: Loc,
) {
    companion object {
        fun parse(arr: List<JSON.Element>): Span =
            Span(
                arr[0].str,
                Loc.parse(arr[1].arr),
                Loc.parse(arr[2].arr)
            )

        fun parseNew(str: String): Span {
            val (_, file, rem) = str.translateEscapes().split('"', limit = 3)
            val remr = rem.split(' ')
            return Span(
                file,
                Loc.parse(JSON.parse(remr[1])!!.arr),
                Loc.parse(JSON.parse(remr[2])!!.arr)
            )
        }
    }
}

@JvmInline
value class SpanRef(
    val index: List<Int>
) {
    override fun toString(): String =
        "SpanRef"
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
    var value: Either<String, Span>,
    var children: List<Instr>,
    val signature: Signature,
    val loc: InstSpan,
    val rec: Boolean,
) {
    override fun toString(): String =
        "Function($signature) { ${children.joinToString()} }"
}

abstract class Instr

data class PrimitiveInstr(
    var id: String,
    var loc: SpanRef? = null,

    val param: Int? = null,
    val typeParam: Type? = null,
): Instr() {
    override fun toString(): String =
        "PrimitiveInstr($id:$param)"
}

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
): ImmInstr() {
    companion object {
        fun parse(arr: List<JSON.Element>): PushFnInstr {
            val value: Either<String, Span> =
                if (arr[0].isStr()) {
                    Either.ofA(arr[0].str)
                } else {
                    Either.ofB(Span.parse(arr[0].arr))
                }
            val signature = Signature.parse(arr[1].arr)
            val loc = InstSpan.parse(arr[2].arr)
            return PushFnInstr(Function(
                value,
                listOf(),
                signature,
                loc,
                if (arr[4].num == 0.0) false else true
            ))
        }
    }
}

data class PushFnRefInstr(
    var fn: String
): ImmInstr()

data class CopyTempStackInstr(
    val stack: String,
    val count: Int,
): Instr()

data class PushTempStackInstr(
    val stack: String,
    val count: Int,
): Instr()

data class PopTempStackInstr(
    val stack: String,
    val count: Int,
): Instr()

data class CommentInstr(
    val comment: String
): Instr()

data class FlagInstr(
    val flag: String
): Instr()

data class SourceLocInstr(
    val uasmSpanIdc: List<Int>,
): Instr()