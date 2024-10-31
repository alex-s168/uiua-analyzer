package me.alex_s168.uiua.ir

fun MutableList<IrVar>.updateVar(old: IrVar, new: IrVar) {
    repeat(size) { index ->
        val irVar = this[index]
        if (irVar.id == old.id)
            this[index] = new
    }
}

fun MutableMap<String, IrBlock>.putBlock(block: IrBlock) {
    this[block.name] = block
}