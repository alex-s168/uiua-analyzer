package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerPervasive(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.ADD,
                Prim.SUB,
                Prim.MUL,
                Prim.DIV,
                Prim.LT,
                Prim.EQ,
                Prim.POW -> {
                    val outTy = instr.outs[0].type
                    if (outTy is ArrayType) {
                        val aTy = instr.args[0].type.let { if (it is ArrayType) it.inner else it }
                        val bTy = instr.args[1].type.let { if (it is ArrayType) it.inner else it }

                        var idx = instrs.indexOf(instr)
                        instrs.removeAt(idx)

                        instrs.add(idx ++, comment("+++ $instr"))

                        val fn = IrBlock(anonFnName(), ref).apply {
                            val a = newVar().copy(type = aTy).also { args += it }
                            val b = newVar().copy(type = bTy).also { args += it }

                            val res = newVar().copy(type = outTy.inner).also { rets += it }

                            instrs += IrInstr(
                                mutableListOf(res),
                                PrimitiveInstr(instr.instr.id),
                                mutableListOf(a, b)
                            )

                            putBlock(this)
                        }

                        val fnFn = newVar().copy(type = fn.type())
                        instrs.add(idx ++, IrInstr(
                            mutableListOf(fnFn),
                            PushFnRefInstr(fn.name),
                            mutableListOf()
                        ))

                        instrs.add(idx ++, IrInstr(
                            mutableListOf(instr.outs[0]),
                            PrimitiveInstr(Prim.EACH),
                            mutableListOf(fnFn, instr.args[0], instr.args[1])
                        ))

                        instrs.add(idx ++, comment("--- $instr"))
                    }
                }
            }
        }
    }
}