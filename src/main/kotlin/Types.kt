package me.alex_s168.uiua

import blitz.collections.contents
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
    val of: Type
): Type("box", listOf()) {
    override fun toString(): String =
        "box[$of]"
}

fun Type.combine(other: Type): Type =
    when (this) {
        Types.byte -> other
        Types.int -> when (other) {
            Types.byte,
            Types.int -> Types.int
            Types.double -> Types.double
            else -> error("")
        }
        Types.double -> Types.double
        else -> error("")
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

    val inner by lazy {
        var curr: Type = this
        while (curr is ArrayType) {
            val currArr = curr as ArrayType
            curr = currArr.of
        }
        curr
    }

    fun mapInner(fn: (Type) -> Type): ArrayType =
        shape.shapeToType(fn(inner))

    override fun toString(): String =
        "arr[$of]${length ?: "?"}"
}

// TODO: verify
fun List<Int>.shapeCompact(): List<Int> =
    takeWhile { it != 0 }.dropLastWhile { it in arrayOf(1, -1) }

fun List<Int>.shapeEq(other: List<Int>): Boolean =
    this.shapeCompact().contents == other.shapeCompact().contents

fun List<Int>.shapeToType(elem: Type): ArrayType {
    val left = toMutableList()
    var arr = ArrayType(elem, left.removeLast().let { if (it < 0) null else it })
    while (left.isNotEmpty())
        arr = ArrayType(arr, left.removeLast().let { if (it < 0) null else it })
    return arr
}

class PtrType(
    val to: Type
): Type("ptr", listOf()) {
    override fun toString(): String =
        "ptr[$to]"
}

fun Type.isAllocated(): Boolean =
    when (this) {
        is ArrayType -> true
        is BoxType -> true
        is PtrType -> false
        is NumericType -> false
        Types.dynamic -> false
        Types.func -> false
        Types.opaque -> false
        else -> TODO()
    }

object Types {
    val tbd = object : Type("tbd", listOf()) {}

    /* numeric */
    val double = NumericType("float", listOf())
    val int = NumericType("int", listOf(double))
    val byte = NumericType("byte", listOf(int, double))

    /* general */
    val dynamic = Type("dyn", listOf())
    fun box(of: Type) = BoxType(of)
    val func = Type("func", listOf())

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
            "func" -> {
                require(arg.isEmpty() && arg2.isEmpty())
                func
            }
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
                val of = parse(arg)
                require(arg2.isEmpty())
                box(of)
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

/*
# +++ for compatiblity with interpreter
dynamic ← ()
fnsig ← (poppop)
attrib ← (pop)
use ← (unboxbox)
# ---

DoShit ← dynamic(+1) # (this has no reason to be dynamic)
DoShitWrapper ← fnsig["int"]"arr[int]" dynamic(
  DoShit
)

# this is now automatically converted to a template
Inc ← (+1)
IncB ← attrib["inline"](+1)

A ← fnsig["int"]"int"(Inc)
B ← fnsig["int"]"int"(IncB)
C ← fnsig["float"]"float"(Inc)
D ← fnsig["float"]"float"(IncB)
E ← (Inc)
F ← fnsig["int"]"int"(E)
G ← fnsig["dyn"]"int"(use)
H ← fnsig["int"]"int"(G)

# The generated binary should behave somewhat like this:
#
# IntArray DoShitWrapper(int a) {
#   auto interpr = new Interpreter(DoShit__assembly); // pass bytecode to interpreter
#   interpr.push(new Dynamic(a));
#   interpr.run();
#   return interpr.pop().as<IntArray>(); // as throws a runtime exception if unsuccesfull
# }
# int Inc__int(int a) { return a + 1; }
# int Inc__float(double a) { return a + 1; }
# int A(int a) { return Inc__int(a); }
# int B(int a) { return a + 1; }
# float C(float a) { return Inc__float(a); }
# float D(float a) { return a + 1; }
# int E__int(int a) { return Inc__int(a); }
# int F(int a) { return E__int(a); }
# Dynamic G(int a) { return new Dynamic(a); }
# int H(int a) { return G(a).as<IntArray>(); } // as throws a runtime exception if unsuccesfull
 */