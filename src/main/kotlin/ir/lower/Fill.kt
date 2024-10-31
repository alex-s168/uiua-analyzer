package me.alex_s168.uiua.ir.lower

import blitz.Obj
import blitz.collections.findCommon
import me.alex_s168.uiua.FnType
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.withoutParallel
import me.alex_s168.uiua.log

val lowerFill = Pass<Unit>("lower fill") { block, _ ->
    val a = Analysis(block)

    block.fillArg?.let {
        log("added fill arg $it to ${block.name}")
        block.args.add(it)
    }

    block.instrs.toList().forEach { inst ->
        if (a.isPrim(inst, Prims.FILL)) {
            var idx = block.instrs.indexOf(inst)
            val calling = inst.args[1]
            val callingType = calling.type as FnType

            val fillv = if (callingType.fillType != null) {
                val fillv = block.newVar().copy(type = callingType.fillType!!)
                block.instrs.add(idx ++, IrInstr(
                    mutableListOf(fillv),
                    PrimitiveInstr(Prims.CALL),
                    mutableListOf(inst.args[0])
                ))
                listOf(fillv)
            } else listOf()

            block.instrs[idx] = IrInstr(
                inst.outs.toMutableList(),
                PrimitiveInstr(Prims.CALL),
                (listOf(calling) + inst.args.drop(2) + fillv).toMutableList()
            )
        }
        else if (a.isPrim(inst, Prims.SWITCH)) {
            val dests = a.origin(inst.args[1])!!.args
                .map { it.type as FnType }
            val ft = dests.map { Obj.of(it.fillType) }.findCommon()
            require(ft != null) { "all functions called in switch have to take same fill type" }
            ft.v?.let {
                inst.args.add(block.fillArg
                    ?: error("need fill value because calling function that needs fill value"))
            }
        }
        else {
            a.getCalling(inst)?.let {
                val type = inst.args[it].type as FnType
                type.fillType?.let {
                    inst.args.add(block.fillArg
                        ?: error("need fill value because calling function that needs fill value"))
                }
            }
        }
    }

    block.fillArg = null

    block.args.forEach {
        if (it.type is FnType) {
            it.type.fillType = null
        }
    }

    block.instrs.forEach {
        (it.args + it.outs).forEach {
            if (it.type is FnType) {
                it.type.fillType = null
            }
        }
    }
}.withoutParallel()

val fixFnTypes = Pass<Unit>("fix fn types") { block, _ ->
    Analysis(block).updateFnType()
}.withoutParallel()