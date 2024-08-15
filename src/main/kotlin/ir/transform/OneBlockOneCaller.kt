package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.*

val oneBlockOneCaller = Pass<(IrBlock) -> Unit>("1block1caller") { block, putBlock ->
    val a = Analysis(block)

    if (a.callerInstrs().size <= 1)
        return@Pass

    a.callerInstrs().forEach { (callerBlock, caller) ->
        val copy = block.deepCopy()
            .also(putBlock)

        val newFnRef = callerBlock.newVar()
            .copy(type = copy.type())

        callerBlock.instrs.add(
            callerBlock.instrs.indexOf(caller), IrInstr(
                mutableListOf(newFnRef),
                PushFnRefInstr(copy.name),
                mutableListOf()
            )
        )

        Analysis(callerBlock)
            .getDeepCalling(caller)
            .filterCertainlyCalling(callerBlock, block.name)
            .forEach { it.replace(newFnRef) }
    }
}