package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrVar

// TODO: make work trough multiple blocks
fun passValue(from: Pair<IrBlock, IrVar>, to: IrBlock): IrVar? {
    if (from.first == to)
        return from.second

    val fromA = Analysis(from.first)

    from.first.instrs.forEach {
        val calling = fromA.getDeepCalling(it)
            .map { fromA.function(it.get()) }

        if (calling.none { it == null } && calling.any { it == to }) {
            it.args.add(from.second)
            return calling.map {
                it!!
                val v = it.newVar().copy(type = from.second.type)
                it.args += v
                it to v
            }.find { it.first == to }!!.second
        }
    }

    return null
}