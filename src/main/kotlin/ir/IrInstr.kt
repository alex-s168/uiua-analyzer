package me.alex_s168.uiua.ir

import blitz.collections.contents
import blitz.flatten
import me.alex_s168.uiua.*
import kotlin.math.floor

data class IrInstr(
    val outs: MutableList<IrVar>,
    val instr: Instr,
    val args: MutableList<IrVar>,
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

    fun inferTypes(parent: IrBlock, putFn: (IrBlock) -> Unit) {
        fun updateType(variable: IrVar, type: Type) {
            parent.updateVar(variable, variable.copy(type = type))
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
                updateType(outs[0], Types.func)
            }

            is PrimitiveInstr -> when (instr.id) {
                "cUSE" -> {
                    updateType(outs[0], args[0].type)
                }

                "BOX" -> {
                    updateType(outs[0], Types.box(args[0].type))
                }

                "UN_BOX" -> {
                    val ty = args[0].type
                    require(ty is BoxType)
                    require(ty.options.size == 1) // TODO: insert switch
                    updateType(outs[0], ty.options[0])
                }

                "ADD",
                "SUB",
                "MUL",
                "DIV" -> {
                    val ty: Type = args.firstOrNull { it.type is ArrayType }?.let { arr ->
                        arr.type // TODO: problem with arr[int] + arr[double]
                    } ?: args.map { it.type }.reduce { a, b ->
                        a.combine(b)
                    }
                    updateType(outs[0], ty)
                }

                "PRIMES" -> {
                    updateType(outs[0], Types.array(Types.int))
                }

                "EACH" -> {
                    val fn = parent.instrDeclFor(args[0])!!.instr as PushFnRefInstr
                    val inp = args[1].type as ArrayType
                    val fnblock = parent.ref(fn.fn)!!
                    val new = fnblock.expandFor(listOf(inp.inner), putFn)
                    val newb = parent.ref(new)!!
                    val newv = parent.newVar().copy(type = Types.func)
                    parent.instrs.add(
                        parent.instrs.indexOf(this), IrInstr(
                            mutableListOf(newv),
                            PushFnRefInstr(new),
                            mutableListOf(),
                        )
                    )
                    args[0] = newv
                    updateType(outs[0], inp.mapInner { newb.rets[0].type })
                }
            }
        }
    }

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        outs.updateVar(old, new)
    }
}
