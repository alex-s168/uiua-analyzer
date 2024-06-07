package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerRange(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.RANGE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    fun put(instr: IrInstr) {
                        instrs.add(idx ++, instr)
                    }

                    val arrTy = instr.outs[0].type as ArrayType

                    val shape = instr.args.wrapInArgArray(::newVar, ::put)

                    put(IrInstr(
                        instr.outs,
                        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                        mutableListOf(shape)
                    ))

                    val (zero) = constants(::newVar, 0.0, type = Types.int, put = ::put)

                    val fn = IrBlock(anonFnName(), ref, fillArg = fillArg).apply {
                        fillArg?.let { fillArg = newVar().copy(type = it.type) }

                        val index = newVar().copy(type = Types.size).also { args += it }
                        val out = newVar().copy(type = arrTy).also { args += it }

                        val indecies = listOf(index).wrapInArgArray(::newVar, instrs::add)

                        instrs += IrInstr(
                            mutableListOf(),
                            PrimitiveInstr(Prim.Comp.ARR_STORE),
                            mutableListOf(out, indecies, index)
                        )

                        putBlock(this)
                    }

                    val fnRef = newVar().copy(type = fn.type())
                    put(IrInstr(
                        mutableListOf(fnRef),
                        PushFnRefInstr(fn.name),
                        mutableListOf()
                    ))

                    put(IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.REPEAT), // [start], [end], [fn which takes counter], [additional]...
                        mutableListOf(zero, instr.args[0], fnRef, instr.outs[0])
                    ))
                }
            }
        }
    }
}