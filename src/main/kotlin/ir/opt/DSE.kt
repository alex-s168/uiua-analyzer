package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.IrBlock

private fun findBlocks(block: IrBlock, blocks: MutableMap<String, IrBlock>, dest: MutableList<IrBlock>) {
    if (block in dest) return
    dest.add(block)

    block.instrs.forEach {
        if (it.instr is PushFnRefInstr) {
            findBlocks(blocks[it.instr.fn] ?: return, blocks, dest)
        }
    }
}

val dse = { root: String, blocks: MutableMap<String, IrBlock> ->
    val blk = blocks[root]!!
    val relevantBlocks = mutableListOf<IrBlock>()
    findBlocks(blk, blocks, relevantBlocks)

    blocks.toList().forEach { (k, block) ->
        if (block !in relevantBlocks) {
            blocks.remove(k)
        }
    }
}