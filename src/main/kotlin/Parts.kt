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
    }
}

@JvmInline
value class SpanRef(
    val index: Int
) {
    fun resolve(asm: Assembly): Span =
        asm.spans[index]

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
    val value: Either<String, Span>,
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
    val id: String,
    val loc: SpanRef? = null
): Instr() {
    override fun toString(): String =
        "PrimitiveInstr($id)"
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
                arr[4].bool
            ))
        }
    }
}

data class PushFnRefInstr(
    val fn: String
): ImmInstr()

data class CopyTempStackInstr(val stack: String): Instr()
data class PopTempStackInstr(val stack: String): Instr()

data class CommentInstr(
    val comment: String
): Instr()

data class FlagInstr(
    val flag: String
): Instr()