package me.alex_s168.uiua.ir.opt

import blitz.collections.contents
import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.IrBlock

private fun findBlocks(block: IrBlock, blocks: MutableMap<String, IrBlock>, dest: MutableList<IrBlock>) {
    if (block in dest) return
    dest.add(block)

    block.instrs.forEach {
        if (it.instr is PushFnRefInstr) {
            findBlocks(blocks[it.instr.fn]
                ?: error("couldn't find \"${it.instr.fn}\"in ${blocks.keys.contents}"), blocks, dest)
        }
    }
}

val dse = { roots: List<String>, blocks: MutableMap<String, IrBlock> ->
    val relevantBlocks = mutableListOf<IrBlock>()

    roots.forEach {
        findBlocks(blocks[it]!!, blocks, relevantBlocks)
    }

    blocks.toList().forEach { (k, block) ->
        if (block !in relevantBlocks) {
            blocks.remove(k)
        }
    }
}