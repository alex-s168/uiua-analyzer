package me.alex_s168

import blitz.Either
import blitz.collections.*
import blitz.io.Path
import blitz.io.read
import kotlinx.serialization.json.*

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
        fun parse(jsonObject: JsonObject): Assembly {
            val instrs = mutableListOf<Instr>()
            jsonObject["instrs"]!!
                .jsonArray
                .forEach { elem ->
                    val it = (elem as? JsonArray)?.let { PrimitiveInstr(it[0].jsonPrimitive.content, it[1].jsonPrimitive.int) }
                        ?: (elem as JsonObject).let {
                            it["d"]?.let { d -> NumImmInstr(d.jsonArray[0].jsonPrimitive.double) }
                                ?: it["PushFunc"]?.jsonArray?.let { p ->
                                    PushFnInstr(Function(
                                        (p[0] as? JsonPrimitive)?.let { Either.ofA(it.content) } ?: Either.ofB(SpanRef()),
                                        listOf(),
                                        p[1].jsonArray.let { Signature(it[0].jsonPrimitive.int, it[1].jsonPrimitive.int) },
                                        p[2].jsonArray[0].jsonPrimitive.int,
                                        p[2].jsonArray[1].jsonPrimitive.int,
                                        p[3].jsonPrimitive.boolean
                                    ))
                                }
                                ?: it["Comment"]?.let { CommentInstr(it.jsonPrimitive.content) }
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

        fun parse(jsonElement: JsonElement): Assembly =
            parse(jsonElement.jsonObject)

        fun parse(json: String): Assembly =
            parse(Json.parseToJsonElement(json))
    }
}

fun main() {
    val test = Path.of("test.uasm")
        .getFile()
        .read()
        .stringify()
        .flatten()
    val assembly = Assembly.parse(test)
    println(assembly.instructions)
}