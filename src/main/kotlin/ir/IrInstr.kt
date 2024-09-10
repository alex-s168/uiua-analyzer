package me.alex_s168.uiua.ir

import blitz.collections.contents
import blitz.flatten
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ast.AstNode
import kotlin.math.floor

/* funny type inference algo:
var o: Type? = null
var i = Types.tbd
var newb: IrBlock? = null
var new = "aaaaaaaaa"
while (o != i) {
    i = i.cycle()

    newb = kotlin.runCatching {
        new = fnblock.expandFor(listOf(i, inp.of), putFn, fillType)
        parent.ref(new)!!
    }.getOrNull()

    o = newb?.rets?.getOrNull(0)?.type
}
newb!!
 */

data class IrInstr(
    val outs: MutableList<IrVar>,
    val instr: Instr,
    var args: MutableList<IrVar>,

    val ast: AstNode? = null
) {
    override fun toString(): String {
        val res = StringBuilder()

        if (outs.size > 0) {
            outs.forEachIndexed { index, irVar ->
                if (index > 0)
                    res.append(", ")
                res.append(irVar)
            }
            res.append(" = ")
        }

        val innerStr = when (instr) {
            is PrimitiveInstr -> "${instr.id}${instr.param?.let { "($it)" } ?: ""}${ (if (uacPrintSpans) instr.loc?.index?.contents?.let { "@$it" } else null) ?: "" }"
            is ArrImmInstr -> "arr-make ${instr.values.flatten().contents}"
            is NumImmInstr -> "imm ${instr.value}"
            is PushFnInstr -> "fn-make ${instr.fn}"
            is CommentInstr -> "# ${instr.comment}"
            is FlagInstr -> "flag ${instr.flag}"
            is PushFnRefInstr -> "fn ${instr.fn}"
            is SourceLocInstr -> "debug ${instr.uasmSpanIdc.contents}"
            else -> instr::class.simpleName
        }
        res.append(innerStr)

        args.forEach {
            res.append(", ")
            res.append(it)
        }

        res.append(";")

        return res.toString()
    }

    fun deepCopy(): IrInstr =
        copy(
            outs = outs.toMutableList(),
            instr = instr,
            args = args.toMutableList()
        )

    internal fun inferTypes(
        parent: IrBlock,
        putFn: (IrBlock) -> Unit,
        fillType: Type? = null
    ) {
        fun updateType(variable: IrVar, type: Type) {
            parent.updateVar(variable, variable.copy(type = type))
        }

        fun fnRef(to: String): IrVar {
            val fn = parent.ref[to]!!
            val newv = parent.newVar().copy(type = fn.type())
            parent.instrs.add(
                parent.instrs.indexOf(this), IrInstr(
                    mutableListOf(newv),
                    PushFnRefInstr(to),
                    mutableListOf(),
                )
            )
            return newv
        }

        fun highestShapeType(a: ArrayType, b: ArrayType): ArrayType {
            val asl = a.shape.size
            val bsl = b.shape.size

            if (asl > bsl)
                return a

            if (bsl > asl)
                return b

            return a.combineShape(b)
        }

        fun highestShapeType(a: Type, b: Type) =
            when (a) {
                is ArrayType -> when (b) {
                    is ArrayType -> highestShapeType(a, b)
                    else ->  a
                }
                else -> b
            }

        when (instr) {
            is ArrImmInstr -> {
                updateType(outs[0], instr.type.copy(vaOff = true))
            }

            is NumImmInstr -> {
                val ty = if (floor(instr.value) == instr.value) {
                    if (instr.value in 0.0..255.0) Types.autobyte
                    else Types.int
                } else Types.double
                updateType(outs[0], ty)
            }

            is PushFnRefInstr -> {
                val fn = parent.ref[instr.fn]!!
                updateType(outs[0], fn.type())
            }

            is PrimitiveInstr -> when (instr.id) {
                Prim.Comp.USE -> {
                    updateType(outs[0], args[0].type)
                }

                Prim.Comp.BOX_LOAD -> {
                    val box = args[0]
                    val boxType = box.type as BoxType
                    updateType(outs[0], boxType.of)
                }

                Prim.Comp.BOX_STORE -> {
                    val box = args[0]
                    val boxType = box.type as BoxType
                    val value = args[1]
                    require(value.type == boxType.of)
                }

                Prim.Comp.BOX_CREATE -> {
                    val box = outs[0]
                    val boxType = box.type as BoxType

                    if (boxType == Types.tbd) {
                        val usage = parent.instrs.firstOrNull {
                            it.instr is PrimitiveInstr &&
                            it.instr.id == Prim.Comp.BOX_STORE &&
                            box in it.args
                        } ?: error("untyped box not allowed!")

                        val storedVal = usage.args[1]

                        updateType(box, Types.box(storedVal.type))
                    }
                }

                Prim.Comp.BOX_DESTROY -> {}

                Prim.Comp.ARG_ARR -> {
                    updateType(outs[0], Types.array(args[0].type))
                }

                Prim.Comp.ARR_MATERIALIZE -> {
                    updateType(outs[0], (args[0].type as ArrayType).copy(vaOff = true))
                }

                Prim.Comp.ARR_ALLOC -> {
                    require(outs[0].type != Types.tbd)
                }

                Prim.Comp.ARR_STORE -> {}

                Prim.Comp.ARR_LOAD -> {
                    val arrType = args[0].type as ArrayType
                    val indecies = parent.instrDeclFor(args[1])!!
                    val elemType = arrType.into(indecies.args.size)
                    updateType(outs[0], elemType)
                }

                Prim.Comp.ARR_DESTROY -> {}

                Prim.Comp.REPEAT -> {}

                Prim.Comp.DIM -> {
                    updateType(outs[0], Types.size)
                }

                Prim.DESHAPE -> {
                    val arrTy = args[0].type as ArrayType
                    updateType(outs[0], Types.array(arrTy.inner))
                }

                Prim.CALL -> {
                    val (_, fn) = parent.funDeclFor(args[0])!!
                    val callArgs = args.drop(1)

                    val exp = fn.expandFor(callArgs.map { it.type }, putFn, fillType)
                    val expb = parent.ref[exp]!!

                    outs.zip(expb.rets).forEach { (a, b) ->
                        updateType(a, b.type)
                    }
                }

                Prim.BOX -> {
                    updateType(outs[0], Types.box(args[0].type))
                }

                Prim.UN_BOX -> {
                    val ty = args[0].type as BoxType
                    updateType(outs[0], ty.of)
                }

                Prim.POW,
                Prim.DIV -> {
                    val arrTy = args.firstOrNull { it.type is ArrayType }?.type as ArrayType?
                    updateType(outs[0], arrTy?.mapInner { Types.double } ?: Types.double)
                }

                Prim.ABS -> {
                    updateType(outs[0], args[0].type)
                }

                Prim.RAND,
                Prim.REPLACE_RAND,
                Prim.SIN,
                Prim.NOW -> {
                    updateType(outs[0], Types.double)
                }

                Prim.EQ,
                Prim.LT,
                Prim.ADD,
                Prim.SUB,
                Prim.MUL,
                Prim.MOD -> {
                    val ty: Type = (args.firstOrNull { it.type is ArrayType }?.let { arr ->
                        arr.type.copyType() // TODO: problem with arr[int] + arr[double]
                    } ?: args.map { it.type.copyType() }.reduce { a, b ->
                        a.combine(b)
                    }).let {
                        // we don't want inaccuracy
                        if (it is AutoByteType) Types.int
                        else it
                    }
                    updateType(outs[0], ty)
                }

                Prim.PRIMES -> {
                    updateType(outs[0], Types.array(Types.int))
                }

                Prim.RANGE -> {
                    updateType(outs[0], Types.array(Types.int))
                }

                Prim.DUP -> {
                    val ty = args[0].type
                    updateType(outs[0], ty)
                    updateType(outs[1], ty)
                }

                Prim.FLIP -> {
                    updateType(outs[0], args[1].type)
                    updateType(outs[1], args[0].type)
                }

                Prim.OVER -> {
                    updateType(outs[0], args[1].type)
                    updateType(outs[1], args[0].type)
                    updateType(outs[2], args[1].type)
                }

                Prim.LEN -> {
                    updateType(outs[0], Types.size)
                }

                Prim.SWITCH -> {
                    val at = parent.instrDeclFor(args[0])!!
                    require(at.instr is PrimitiveInstr && at.instr.id == Prim.Comp.ARG_ARR)

                    val fnsa = parent.instrDeclFor(args[1])!!
                    require(fnsa.instr is PrimitiveInstr && fnsa.instr.id == Prim.Comp.ARG_ARR)
                    val fns = fnsa.args

                    val on = args[2]
                    require(on.type is NumericType)

                    val inputs = args.drop(3)

                    val expanded = fns.mapIndexed { index, arg ->
                        val (_, fn) = parent.funDeclFor(arg)!!
                        val exp = fn.expandFor(
                            inputs.map { it.type },
                            putFn,
                            fillType
                        )
                        val expb = parent.ref[exp]!!
                        fns[index] = fnRef(exp)
                        expb
                    }

                    require(expanded.all { it.rets.size == outs.size })

                    outs.zip(expanded.first().rets).forEach { (v, t) ->
                        updateType(v, t.type)
                    }
                }

                Prim.REDUCE, Prim.Front.REDUCE_DEPTH -> {
                    val (_, fnblock) = parent.funDeclFor(args[0])!!
                    val inp = args[1].type as ArrayType

                    val depth = if (instr.id == Prim.Front.REDUCE_DEPTH) instr.param!! else 0
                    val inpOf = inp.into(depth + 1)

                    val first = fnblock.expandFor(listOf(inpOf, inpOf), putFn, fillType)
                    val firstb = parent.ref[first]!!
                    val accTy = firstb.rets[0].type

                    args[0] = fnRef(first)

                    val all = fnblock.expandFor(listOf(accTy, inpOf), putFn, fillType)
                    val allb = parent.ref[all]!!
                    require(allb.rets[0].type == accTy)

                    args.add(fnRef(all))

                    updateType(outs[0], List(depth){-1}.shapeToType(accTy))
                }

                Prim.EACH -> {
                    val (_, fnblock) = parent.funDeclFor(args[0])!!

                    val highestRank = args.drop(1)
                        .filter { it.type is ArrayType && it.type.length != 1 }
                        .maxBy { (it.type as ArrayType).shape.size }
                        .type as ArrayType

                    val inps = args.drop(1).map {
                        if (it.type !is ArrayType || it.type.length == 1) it.type
                        else it.type.inner
                    }
                    val new = fnblock.expandFor(inps, putFn, fillType)
                    val newb = parent.ref[new]!!

                    args[0] = fnRef(new)

                    outs.zip(newb.rets).forEach { (out, ret) ->
                        updateType(out, highestRank.mapInner { ret.type }.copyType())
                    }
                }

                Prim.SHAPE -> {
                    val arg = args[0].type as ArrayType
                    updateType(outs[0], Types.array(Types.size, arg.shape.size))
                }

                Prim.ROWS -> {
                    val (_, fnblock) = parent.funDeclFor(args[0])!!

                    val inps = args.drop(1).map {
                        if (it.type !is ArrayType || it.type.length == 1) it.type
                        else it.type.of.makeVaOffIfArray()
                    }

                    val new = fnblock.expandFor(inps, putFn, fillType)
                    val newb = parent.ref[new]!!

                    args[0] = fnRef(new)

                    outs.zip(newb.rets).forEach { (out, ret) ->
                        updateType(out, Types.array(ret.type))
                    }
                }

                Prim.WHERE -> {
                    val arrTy = args[0].type as ArrayType
                    val shaLen = arrTy.shape.size

                    val ty = if (shaLen == 1) Types.array(Types.size)
                    else Types.array(Types.array(Types.size, length = shaLen))

                    updateType(outs[0], ty)
                }

                Prim.FILL -> {
                    val (_, fillValFn) = parent.funDeclFor(args[0])!!
                    val (_, opFn) = parent.funDeclFor(args[1])!!
                    val opArgs = args.drop(2)

                    val fillValFnExp = fillValFn.expandFor(listOf(), putFn, fillType)
                    val fillValFnExpBlock = parent.ref[fillValFnExp]!!
                    val opFnExp = opFn.expandFor(opArgs.map { it.type }, putFn, fillValFnExpBlock.rets[0].type)

                    args[0] = fnRef(fillValFnExp)
                    args[1] = fnRef(opFnExp)

                    outs.zip(parent.ref[opFnExp]!!.rets).forEach { (out, ret) ->
                        updateType(out, ret.type)
                    }
                }

                Prim.FIX -> {
                    val argTy = args[0].type
                    updateType(outs[0], Types.array(argTy, 1))
                }

                Prim.REVERSE -> {
                    val argTy = args[0].type
                    if (argTy is BoxType) {
                        require(argTy.of is ArrayType)
                    } else {
                        require(argTy is ArrayType)
                    }
                    updateType(outs[0], argTy.copyType())
                }

                Prim.PICK -> {
                    val arr = args[1].type as ArrayType
                    updateType(outs[0], arr.of.makeVaOffIfArray())
                }

                Prim.UNDO_PICK -> {
                    val arr = args[0].type as ArrayType
                    updateType(outs[0], arr)
                }

                Prim.TABLE -> {
                    val (_, fn) = parent.funDeclFor(args[0])!!
                    val arr0 = args[1].type as ArrayType
                    val arr1 = args[2].type as ArrayType

                    val fnExp = fn.expandFor(listOf(arr0.of, arr1.of), putFn, fillType)
                    val fnExpBlock = parent.ref[fnExp]!!

                    outs.zip(fnExpBlock.rets).forEach { (out, ret) ->
                        updateType(out, Types.array(Types.array(ret.type)))
                    }
                }

                Prim.RESHAPE -> {
                    val sha = args[0]

                    if (sha.type is ArrayType) {
                        val arr = args[1]
                        val innerTy = if (arr.type is ArrayType) arr.type.inner else arr.type

                        require(sha.type.length != null) {
                            "length of shape has to be known at comptime!"
                        }

                        val resTy = List(sha.type.length) { -1 }.shapeToArrType(innerTy)
                        updateType(outs[0], resTy)
                    }
                    else { // copy the value as rows of new array
                        val inner = args[1].type
                        val resTy = Types.array(inner)
                        updateType(outs[0], resTy)
                    }
                }

                Prim.UN_SHAPE -> {
                    val sha = args[0]
                    val shaTy = sha.type as ArrayType

                    require(shaTy.length != null) {
                        "length of shape has to be known at comptime!"
                    }

                    val resTy = List(shaTy.length) { -1 }.shapeToArrType(Types.int)

                    updateType(outs[0], resTy)
                }

                Prim.JOIN -> {
                    val arg0 = args[0].type
                    val arg1 = args[1].type

                    val res = highestShapeType(arg0, arg1)
                    updateType(outs[0], res)
                }

                Prim.KEEP -> {
                    updateType(outs[0], args[1].type)
                }

                Prim.COMPLEX -> {
                    updateType(outs[0], Types.complex)
                }

                else -> error("infer type for ${instr.id} not implemented")
            }

            is SourceLocInstr -> {
                require(outs.size == 0)
            }
        }

        require(outs.none { it.type == Types.tbd }) {
            "type not inferred in op: $this"
        }
    }

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        outs.updateVar(old, new)
    }
}
