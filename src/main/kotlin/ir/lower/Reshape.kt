package me.alex_s168.uiua.ir.lower

import blitz.collections.contents
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.wrapInArgArray
import me.alex_s168.uiua.ir.withPassArg

val lowerReshape = withPassArg<(IrBlock) -> Unit>("lower reshape") { putBlock ->
    lowerPrimPass(Prim.RESHAPE) { put, newVar, a ->
        if (args[0].type !is ArrayType) { // => scalar
            // repeat array as rows of new array
            val oldValue = args[1]
            val oldValueType = oldValue.type
            val out = outs[0]

            val shape = mutableListOf(args[0])
            if (oldValueType is ArrayType) {
                shape += List(oldValueType.shape.size) {
                    val (d) = constants(newVar, it.toDouble(), type = Types.size, put = put)

                    val dim = newVar().copy(type = Types.size)
                    put(IrInstr(
                        mutableListOf(dim),
                        PrimitiveInstr(Prim.Comp.DIM),
                        mutableListOf(oldValue, d)
                    ))
                    dim
                }
            }

            put(IrInstr(
                mutableListOf(out),
                PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                mutableListOf(shape.wrapInArgArray(newVar, put = put))
            ))

            val loopBody = IrBlock(anonFnName(), a.block.ref).apply {
                val iteration = newVar().copy(type = Types.size).also(args::add)
                val out = newVar().copy(type = out.type).also(args::add)
                val oldValue = newVar().copy(type = oldValueType).also(args::add)

                val idx = listOf(iteration)
                    .wrapInArgArray(newVar, put = instrs::add)

                instrs += IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prim.Comp.ARR_STORE),
                    mutableListOf(out, idx, oldValue)
                )

                putBlock(this)
            }

            val fnref = newVar().copy(type = loopBody.type())
            put(IrInstr(
                mutableListOf(fnref),
                PushFnRefInstr(loopBody.name),
                mutableListOf()
            ))

            val (zero) = constants(newVar, 0.0, type = Types.size, put = put)

            put(IrInstr(
                mutableListOf(),
                PrimitiveInstr(Prim.Comp.REPEAT),
                mutableListOf(zero, args[0], fnref, out, oldValue)
            ))
        }
        else {
            val newShape = a.origin(args[0])!!.args.toList()
            val oldValue = args[1]

            // if have too many elems
            //  truncate
            //
            // if have too few elems:
            //  if have fill value:
            //   use fill value to extend
            //  else:
            //   cycle trough old values

            TODO("not implemented yet")
        }
    }
}