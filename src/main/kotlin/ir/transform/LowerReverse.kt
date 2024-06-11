package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerReverse(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.FIX -> { // TODO: move somewhere else
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(instr.outs[0]),
                        PrimitiveInstr(Prim.BOX),
                        mutableListOf(instr.args[0])
                    ))
                }

                Prim.PICK -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val at = instr.args[0]
                    val arr = instr.args[1]

                    val idc = listOf(at).wrapInArgArray(::newVar) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(instr.outs[0]),
                        PrimitiveInstr(Prim.Comp.ARR_LOAD),
                        mutableListOf(arr, idc)
                    ))
                }

                Prim.UNDO_PICK -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val at = instr.args[0]
                    val arr = instr.args[1]
                    val value = instr.args[2]

                    instrs.add(idx ++, comment("+++ undo_pick"))

                    val oldlen = newVar().copy(type = Types.size)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(oldlen),
                        PrimitiveInstr(Prim.LEN),
                        mutableListOf(arr)
                    ))

                    val sha = listOf(oldlen).wrapInArgArray(::newVar) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(instr.outs[0]),
                        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                        mutableListOf(sha)
                    ))

                    val idc = listOf(at).wrapInArgArray(::newVar) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.ARR_STORE),
                        mutableListOf(instr.outs[0], idc, value)
                    ))

                    instrs.add(idx ++, comment("--- undo_pick"))
                }

                Prim.REVERSE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    instrs.add(idx ++, comment("+++ reverse"))

                    val arg = instr.args[0]

                    val operate = if (arg.type is BoxType) {
                        val unboxed = newVar().copy(type = arg.type.of)
                        instrs.add(idx ++, IrInstr(
                            mutableListOf(unboxed),
                            PrimitiveInstr(Prim.UN_BOX),
                            mutableListOf(arg)
                        ))
                        unboxed
                    } else arg
                    val operateTy = operate.type as ArrayType

                    val out = if (arg.type is BoxType) {
                        newVar().copy(type = arg.type.of)
                    } else instr.outs[0]
                    val outTy = out.type

                    val len = newVar().copy(type = Types.size)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(len),
                        PrimitiveInstr(Prim.LEN),
                        mutableListOf(operate),
                    ))

                    val shape = listOf(len).wrapInArgArray(::newVar) { instrs.add(idx ++, it) }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(out),
                        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                        mutableListOf(shape)
                    ))
                    
                    val block = IrBlock(anonFnName(), ref).apply {
                        val idx = newVar().copy(type = Types.size).also { args += it }
                        val operate = newVar().copy(type = operateTy).also { args += it }
                        val out = newVar().copy(type = outTy).also { args += it }
                        val lenM1 = newVar().copy(type = Types.size).also { args += it }

                        val indec = listOf(idx).wrapInArgArray(::newVar) { instrs += it }

                        val temp = newVar().copy(type = operateTy.of)
                        instrs += IrInstr(
                            mutableListOf(temp),
                            PrimitiveInstr(Prim.Comp.ARR_LOAD),
                            mutableListOf(operate, indec)
                        )

                        val revIdx = newVar().copy(type = Types.size)
                        instrs += IrInstr(
                            mutableListOf(revIdx),
                            PrimitiveInstr(Prim.SUB),
                            mutableListOf(idx, lenM1)
                        )

                        val revIdcs = listOf(revIdx).wrapInArgArray(::newVar) { instrs += it }

                        instrs += IrInstr(
                            mutableListOf(),
                            PrimitiveInstr(Prim.Comp.ARR_STORE),
                            mutableListOf(out, revIdcs, temp)
                        )

                        putBlock(this)
                    }

                    val (zero, one) = constants(::newVar, 0.0, 1.0, type = Types.size) {
                        instrs.add(idx ++, it)
                    }

                    val blockFn = newVar().copy(type = block.type())
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(blockFn),
                        PushFnRefInstr(block.name),
                        mutableListOf()
                    ))

                    val lenM1 = newVar().copy(type = Types.size)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(lenM1),
                        PrimitiveInstr(Prim.SUB),
                        mutableListOf(one, len)
                    ))

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.REPEAT),
                        mutableListOf(zero, len, blockFn, operate, out, lenM1)
                    ))

                    out.into(instr.outs[0]) { instrs.add(idx ++, it) }

                    instrs.add(idx ++, comment("--- reverse"))
                }
            }
        }
    }
}