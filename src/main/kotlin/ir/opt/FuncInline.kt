package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass

val funcInline = Pass<Unit>("Inline") { block, _ ->
    val a = Analysis(block)

    block.instrs.toList().forEach { instr ->
        if (a.isPrim(instr, Prim.CALL)) {
            block.funDeclFor(instr.args[0])?.let { (_, called) ->
                if (called.shouldInline()) {
                    val inlined = called.inlinableCopy(instr.args.drop(1), instr.outs.toList(), block.fillArg)
                    val idx = block.instrs.indexOf(instr)
                    block.instrs.removeAt(idx)
                    block.instrs.addAll(idx, inlined.instrs)
                    println("inlined ${inlined.instrs.size} instrs into ${block.name}!")
                }
            }
        }
        // CANT DO THAT BECAUSE OTHER CALLS USE FILLS
        /*else if (a.isPrim(instr, Prim.FILL)) {
            val provider = block.funDeclFor(instr.args[0])?.second ?: return@forEach
            val called = block.funDeclFor(instr.args[1])?.second ?: return@forEach

            val fill = block.newVar().copy(type = provider.rets[0].type)

            val idx = block.instrs.indexOf(instr)
            block.instrs.removeAt(idx)

            val instrs = mutableListOf<IrInstr>()
            instrs += provider.inlinableCopy(listOf(), listOf(fill)).instrs
            instrs += called.inlinableCopy(instr.args.drop(2), instr.outs.toList(), fill).instrs

            block.instrs.addAll(idx, instrs)
        }*/
    }
}