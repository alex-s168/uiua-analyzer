package me.alex_s168.uiua

import blitz.Either
import blitz.collections.RefVec
import blitz.parse.JSON
import blitz.parse.JSON.asArr
import blitz.parse.JSON.asNum
import blitz.parse.JSON.asStr
import blitz.parse.JSON.uncheckedAsStr

data class Signature(
    val inputs: Int,
    val outputs: Int
) {
    fun mapIns(fn: (Int) -> Int) =
        Signature(fn(inputs), outputs)

    fun mapOuts(fn: (Int) -> Int) =
        Signature(inputs, fn(outputs))

    companion object {
        fun parse(arr: RefVec<JSON.Element>): Signature =
            Signature(
                arr[0].asNum().toInt(),
                arr[1].asNum().toInt()
            )
    }
}

data class Span(
    val sourceFile: String,
    val start: Loc,
    val end: Loc,
) {
    companion object {
        fun parse(arr: RefVec<JSON.Element>): Span =
            Span(
                arr[0].asStr(),
                Loc.parse(arr[1].asArr()),
                Loc.parse(arr[2].asArr())
            )

        fun parseNew(str: String): Span {
            val (_, file, rem) = str.translateEscapes().split('"', limit = 3)
            val remr = rem.split(' ')
            return Span(
                file,
                Loc.parse(JSON.parse(remr[1]).assertA().asArr()),
                Loc.parse(JSON.parse(remr[2]).assertA().asArr())
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
        fun parse(arr: RefVec<JSON.Element>): Loc =
            Loc(
                arr[0].asNum().toInt(),
                arr[1].asNum().toInt(),
                arr[2].asNum().toInt(),
                arr[3].asNum().toInt()
            )
    }
}

data class InstSpan(
    val start: Int,
    val len: Int,
) {
    companion object {
        fun parse(arr: RefVec<JSON.Element>): InstSpan =
            InstSpan(
                arr[0].asNum().toInt(),
                arr[1].asNum().toInt()
            )
    }
}

data class Function(
    var value: Either<String, Span>,
    var children: List<Instr>,
    val signature: Signature?,
    val loc: InstSpan?,
    val rec: Boolean,
) {
    override fun toString(): String =
        "Function($signature) { ${children.joinToString()} }"
}

abstract class Instr

// TODO: make smaller
data class PrimitiveInstr(
    var id: Prim,
    var loc: SpanRef? = null,

    val param: Int? = null,
    val typeParam: Type? = null,
): Instr() {
    override fun toString(): String =
        "PrimitiveInstr(${Prims.all[id]}:$param)"
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
        fun parse(arr: RefVec<JSON.Element>): PushFnInstr {
            val value: Either<String, Span> =
                if (arr[0].kind == JSON.Element.STR) {
                    Either.ofA(arr[0].uncheckedAsStr())
                } else {
                    Either.ofB(Span.parse(arr[0].asArr()))
                }
            val signature = Signature.parse(arr[1].asArr())
            val loc = InstSpan.parse(arr[2].asArr())
            return PushFnInstr(Function(
                value,
                listOf(),
                signature,
                loc,
                if (arr[4].asNum() == 0.0) false else true
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