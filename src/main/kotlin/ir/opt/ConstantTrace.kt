package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.*

val constantTrace = Pass<Unit>("const trace") { block, _ ->
    val a = Analysis(block)

    require(a.callerInstrs().size <= 1) {
        "you forgot to run oneBlockOneCaller pass before constantTrace pass"
    }

    block.instrs.toList().forEach {
        if (a.isPrim(it, Prim.CALL)) {
            val fnv = it.args[0]

            if (block.instrDeclFor(fnv) == null) {
                a.deepOrigin(fnv)?.let { (_, orig) ->
                    if (orig.instr is PushFnRefInstr) {
                        val newv = block.newVar().copy(type = block.ref[orig.instr.fn]!!.type())
                        val push = IrInstr(
                            mutableListOf(newv),
                            PushFnRefInstr(orig.instr.fn),
                            mutableListOf()
                        )
                        block.instrs.add(block.instrs.indexOf(it), push)
                        require(newv.type == fnv.type) { "${newv.type} vs ${fnv.type}" }
                        it.args[0] = newv
                    }
                }
            }
        }
    }
}