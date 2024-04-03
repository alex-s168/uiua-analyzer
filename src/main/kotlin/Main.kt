package me.alex_s168.uiua

import blitz.Either
import blitz.collections.*
import blitz.io.Path
import blitz.io.read
import blitz.parse.JSON
import blitz.str.flattenToString

data class Signature(
    val inputs: Int,
    val outputs: Int
)

// TODO
class SpanRef

data class Function(
    val value: Either<String, SpanRef>,
    val children: List<Instr>,
    val signature: Signature,
    val loc: Int,
    val len: Int,
    val megic: Boolean
)

abstract class Instr

data class PrimitiveInstr(
    val id: String,
    val loc: Int
): Instr()

abstract class ImmInstr: Instr()

data class NumImmInstr(
    val value: Double
): ImmInstr()

data class PushFnInstr(
    val fn: Function
): ImmInstr()

data class CommentInstr(
    val comment: String
): Instr()

data class Assembly(
    val instructions: List<Instr>
) {
    companion object {
        fun parse(jsonObject: Map<String, JSON.Element>): Assembly {
            val instrs = mutableListOf<Instr>()
            jsonObject["instrs"]!!
                .arr
                .forEach { elem ->
                    println(elem)
                    val it = (elem as? JSON.Array)?.let { PrimitiveInstr(it.value[0].str, it.value[1].num.toInt()) }
                        ?: elem.obj.let {
                            it["d"]?.let { d ->
                                // if (it.containsKey("s"))
                                // TODO
                                NumImmInstr(d.arr[0].num)
                            }
                            ?: it["PushFunc"]?.arr?.let { p ->
                                PushFnInstr(Function(
                                    (p[0] as? JSON.Str)?.let { Either.ofA(it.value) } ?: Either.ofB(SpanRef()),
                                    listOf(),
                                    p[1].arr.let { Signature(it[0].num.toInt(), it[1].num.toInt()) },
                                    p[2].arr[0].num.toInt(),
                                    p[2].arr[1].num.toInt(),
                                    p[3].bool
                                ))
                            }
                            ?: it["Comment"]?.let { CommentInstr(it.str) }
                            ?: error("don't know how to parse $it")
                        }
                    if (it is PushFnInstr) {
                        val last = instrs.removeLastInto(it.fn.len)
                        instrs.add(PushFnInstr(it.fn.copy(children = last)))
                    } else {
                        instrs.add(it)
                    }
                }
            return Assembly(instrs)
        }

        fun parse(json: String): Assembly =
            parse(JSON.parse(json)!!.obj)
    }
}

fun main() {
    val test = Path.of("test.uasm")
        .getFile()
        .read()
        .stringify()
        .flattenToString()
    val assembly = Assembly.parse(test)
    println(assembly.instructions)
}