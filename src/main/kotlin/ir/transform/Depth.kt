package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun depth(newVar: () -> IrVar, put: (IrInstr) -> Unit, ref: Map<String, IrBlock>, putBlock: (IrBlock) -> Unit, depth: Int, args: List<IrVar>, blockBuilder: (newVar: () -> IrVar, put: (IrInstr) -> Unit, args: List<IrVar>) -> List<IrVar>): List<IrVar> {
    if (depth == 0)
        return blockBuilder(newVar, put, args)

    val block = IrBlock(anonFnName(), ref).apply {
        val args = args.map {
            newVar()
                .copy(type = it.type.ofIfArray())
                .also { this.args += it }
        }

        this.rets += depth(this::newVar, { this.instrs += it }, ref, putBlock, depth - 1, args, blockBuilder)
    }.also(putBlock)

    val fn = newVar().copy(type = block.type())
    put(IrInstr(
        mutableListOf(fn),
        PushFnRefInstr(block.name),
        mutableListOf()
    ))

    val rets = block.rets
        .map { newVar().copy(type = Types.array(it.type)) }

    put(IrInstr(
        rets.toMutableList(),
        PrimitiveInstr(Prim.ROWS),
        (listOf(fn) + args).toMutableList()
    ))

    return rets
}