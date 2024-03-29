package me.alex_s168

import blitz.str.splitWithNesting

// TODO: do we need equals functions?

open class Type(
    val name: String,
    val implicitConv: List<Type>
) {
    override fun toString(): String =
        name
}

class NumericType(
    name: String,
    down: List<Type>
): Type(name, down)

class BoxType(
    val options: List<Type>
): Type("box", listOf()) {
    override fun toString(): String =
        "box[${options.joinToString(separator = "|")}]"
}

class ArrayType(
    val of: Type,
    val length: Int?
): Type("arr", listOf()) {
    val shape by lazy {
        val sha = mutableListOf<Int>()
        var curr: Type = this
        while (curr is ArrayType) {
            val currArr = curr as ArrayType
            sha += currArr.length ?: -1
            curr = currArr.of
        }
        sha
    }

    override fun toString(): String =
        "arr[$of]${length ?: "?"}"
}

class PtrType(
    val to: Type
): Type("ptr", listOf()) {
    override fun toString(): String =
        "ptr[$to]"
}

object Types {
    /* numeric */
    val double = NumericType("float", listOf())
    val int = NumericType("int", listOf(double))
    val byte = NumericType("byte", listOf(int, double))

    /* general */
    val dynamic = Type("dyn", listOf())
    fun box(vararg of: Type) = BoxType(of.toList())

    /* array */
    fun array(of: Type, length: Int? = null) = ArrayType(of, length)

    /* native */
    fun pointer(to: Type) = PtrType(to)
    /** Can not be deref-ed; only usable in pointers */
    val opaque = Type("opaque", listOf())


    fun parse(str: String): Type {
        val main = str.substringBefore('[')
        val arg = str.substringAfter('[', missingDelimiterValue = "").substringBeforeLast(']')
        val arg2 = str.substringAfterLast(']', missingDelimiterValue = "")

        return when (main) {
            "float" -> {
                require(arg.isEmpty() && arg2.isEmpty())
                double
            }
            "int" -> {
                require(arg.isEmpty() && arg2.isEmpty())
                int
            }
            "byte" -> {
                require(arg.isEmpty() && arg2.isEmpty())
                byte
            }
            "dyn" -> {
                require(arg.isEmpty() && arg2.isEmpty())
                dynamic
            }
            "box" -> {
                val of = arg.splitWithNesting('|', nestUp = '[', nestDown = ']').map { parse(it) }
                require(arg2.isEmpty())
                box(*of.toTypedArray())
            }
            "arr" -> {
                val of = parse(arg)
                val len = arg2.toIntOrNull()
                array(of, len)
            }
            "ptr" -> {
                val to = parse(arg)
                require(arg2.isEmpty())
                pointer(to)
            }
            "opaque" -> {
                require(arg.isEmpty() && arg2.isEmpty())
                opaque
            }
            else -> error("Unknown type $str!")
        }
    }
}

fun main() {
    println(Types.parse(readln()))
}