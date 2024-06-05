package me.alex_s168.uiua.ir

fun MutableList<IrVar>.updateVar(old: IrVar, new: IrVar) {
    forEachIndexed { index, irVar ->
        if (irVar.id == old.id)
            this[index] = new
    }
}

fun MutableMap<String, IrBlock>.putBlock(block: IrBlock) {
    this[block.name] = block
}