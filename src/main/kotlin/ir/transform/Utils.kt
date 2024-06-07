package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun IrInstr.transformInferTypes(parent: IrBlock, putFn: (IrBlock) -> Unit): IrInstr =
    also {
        inferTypes(parent, putFn, parent.fillArg?.type)
    }

fun constantArr(dest: IrVar, newVar: () -> IrVar, vararg const: Double, type: Type = Types.double, put: (IrInstr) -> Unit) {
    val ca = const.mapTo(mutableListOf()) {
        val constant = newVar().copy(type = type)
        put(IrInstr(
            mutableListOf(constant),
            NumImmInstr(it),
            mutableListOf()
        ))
        constant
    }

    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prim.Comp.ARG_ARR),
        ca
    ))
}