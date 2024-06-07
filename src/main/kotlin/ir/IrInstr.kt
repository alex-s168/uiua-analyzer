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
    val args: MutableList<IrVar>,

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
            is PrimitiveInstr -> instr.id
            is ArrImmInstr -> "arr-make ${instr.values.flatten().contents}"
            is NumImmInstr -> "imm ${instr.value}"
            is PushFnInstr -> "fn-make ${instr.fn}"
            is CommentInstr -> "comment ${instr.comment}"
            is FlagInstr -> "flag ${instr.flag}"
            is PushFnRefInstr -> "fn ${instr.fn}"
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
            val fn = parent.ref(to)!!
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

        when (instr) {
            is ArrImmInstr -> {
                updateType(outs[0], instr.type)
            }

            is NumImmInstr -> {
                val ty = if (floor(instr.value) == instr.value) {
                    if (instr.value in 0.0..255.0) Types.byte
                    else Types.int
                } else Types.double
                updateType(outs[0], ty)
            }

            is PushFnRefInstr -> {
                val fn = parent.ref(instr.fn)!!
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
                    updateType(outs[0], args[0].type)
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

                Prim.CALL -> {
                    val (_, fn) = parent.funDeclFor(args[0])
                    val callArgs = args.drop(1)

                    val exp = fn.expandFor(callArgs.map { it.type }, putFn, fillType)
                    val expb = parent.ref(exp)!!

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

                Prim.ADD,
                Prim.SUB,
                Prim.MUL,
                Prim.DIV -> {
                    val ty: Type = args.firstOrNull { it.type is ArrayType }?.let { arr ->
                        arr.type // TODO: problem with arr[int] + arr[double]
                    } ?: args.map { it.type }.reduce { a, b ->
                        a.combine(b)
                    }
                    updateType(outs[0], ty)
                }

                Prim.PRIMES -> {
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
                        val (_, fn) = parent.funDeclFor(arg)
                        val exp = fn.expandFor(
                            inputs.map { it.type },
                            putFn,
                            fillType
                        )
                        val expb = parent.ref(exp)!!
                        fns[index] = fnRef(exp)
                        expb
                    }

                    require(expanded.all { it.rets.size == outs.size })

                    outs.zip(expanded.first().rets).forEach { (v, t) ->
                        updateType(v, t.type)
                    }
                }

                Prim.REDUCE -> {
                    val (_, fnblock) = parent.funDeclFor(args[0])
                    val inp = args[1].type as ArrayType

                    val first = fnblock.expandFor(listOf(inp.of, inp.of), putFn, fillType)
                    val firstb = parent.ref(first)!!
                    val accTy = firstb.rets[0].type

                    args[0] = fnRef(first)

                    val all = fnblock.expandFor(listOf(accTy, inp.of), putFn, fillType)
                    val allb = parent.ref(all)!!
                    require(allb.rets[0].type == accTy)

                    args.add(fnRef(all))

                    updateType(outs[0], accTy)
                }

                Prim.EACH -> {
                    val (_, fnblock) = parent.funDeclFor(args[0])
                    val inp = args[1].type as ArrayType

                    val new = fnblock.expandFor(listOf(inp.inner), putFn, fillType)
                    val newb = parent.ref(new)!!

                    args[0] = fnRef(new)

                    updateType(outs[0], inp.mapInner { newb.rets[0].type })
                }

                Prim.ROWS -> {
                    val (_, fnblock) = parent.funDeclFor(args[0])

                    val inps = args.drop(1).map { arg ->
                        arg.type as ArrayType
                    }
                    val new = fnblock.expandFor(inps.map { it.of }, putFn, fillType)
                    val newb = parent.ref(new)!!

                    args[0] = fnRef(new)

                    outs.zip(newb.rets).forEach { (out, ret) ->
                        updateType(out, Types.array(ret.type))
                    }
                }

                Prim.FILL -> {
                    val (_, fillValFn) = parent.funDeclFor(args[0])
                    val (_, opFn) = parent.funDeclFor(args[1])
                    val opArgs = args.drop(2)

                    val fillValFnExp = fillValFn.expandFor(listOf(), putFn, fillType)
                    val fillValFnExpBlock = parent.ref(fillValFnExp)!!
                    val opFnExp = opFn.expandFor(opArgs.map { it.type }, putFn, fillValFnExpBlock.rets[0].type)

                    args[0] = fnRef(fillValFnExp)
                    args[1] = fnRef(opFnExp)

                    outs.zip(parent.ref(opFnExp)!!.rets).forEach { (out, ret) ->
                        updateType(out, ret.type)
                    }
                }
            }
        }
    }

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        outs.updateVar(old, new)
    }
}
