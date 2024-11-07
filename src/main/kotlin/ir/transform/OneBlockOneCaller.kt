package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.*

// TODO: remove because of comp speed reasons?
val oneBlockOneCaller = GlobalPass<(IrBlock) -> Unit>("1block1caller") { blocks, putBlock ->
    val callers = blocks
        .values
        .associateWith { Analysis(it).callerInstrs() }

    callers.forEach { (block, callers) ->
        callers.drop(1).forEach { (callerBlock, caller) ->
            val copy = block.deepCopy()
                .also(putBlock)

            val newFnRef = callerBlock.newVar()
                .copy(type = copy.type())

            callerBlock.instrs.add(
                callerBlock.instrs.indexOf(caller), IrInstr(
                    mutableListOf(newFnRef),
                    PushFnRefInstr(copy.uid),
                    mutableListOf()
                )
            )

            Analysis(callerBlock)
                .getDeepCalling(caller)
                .filterCertainlyCalling(callerBlock, block.uid)
                .forEach { it.replace(newFnRef) }
        }
    }
}