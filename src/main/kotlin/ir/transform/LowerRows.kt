package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerRows(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.ROWS -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val fn = instr.args[0]
                    val inputs = instr.args.drop(1)

                    val inputsLen = inputs.map {
                        val v = newVar().copy(type = Types.size)
                        instrs.add(idx ++, IrInstr(
                            mutableListOf(v),
                            PrimitiveInstr(Prim.LEN),
                            mutableListOf(it)
                        ))
                        v
                    }

                    // !!! to allocate the array, we first have to run the function once to get the dimension of the inner arrays

                    // execute iter 0
                    // allocate array
                    // copy result from iter 0 into array
                    // free result from iter 0
                    // repeat for remaining iters:
                    //   execute iter i
                    //   copy result from iter i into array
                    //   free result from iter i

                    // TODO
                }
            }
        }
    }
}