import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.switch

fun boundsCheck(block: IrBlock, arr: IrVar, indexes: List<IrVar>, newVar: () -> IrVar, putBlock: (IrBlock) -> Unit, put: (IrInstr) -> Unit) {
    val a = Analysis(block)
    val arrTy = arr.type as ArrayType

    indexes.forEachIndexed { shaIdx, index ->
        val (shaIdxV) = constants(newVar, shaIdx.toDouble(), type = Types.size, put = put)

        val atShape = arrTy.shape[shaIdx]

        if (atShape != -1) {
            a.origin(index)?.instr?.let { idxDecl ->
                if (idxDecl is NumImmInstr) {
                    val idx = idxDecl.value.toInt()
                    if (idx >= atShape) {
                        error("(comptime catched) Index $idx out of bounds for array with size $atShape (in shape)")
                    }
                    return@forEachIndexed
                }
            }
        }

        val dim = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(dim),
            PrimitiveInstr(Prims.Comp.DIM),
            mutableListOf(arr, shaIdxV)
        ))

        val lt = newVar().copy(type = Types.bool)
        put(IrInstr(
            mutableListOf(lt),
            PrimitiveInstr(Prims.LT),
            mutableListOf(dim, index) // index < dim
        ))

        val nop = IrBlock(anonFnName(), a.block.ref).apply {
            putBlock(this)
        }

        val panic = IrBlock(anonFnName(), a.block.ref).apply {
            instrs += IrInstr(
                mutableListOf(),
                PrimitiveInstr(Prims.Comp.PANIC),
                mutableListOf()
            )

            putBlock(this)
        }

        val (zero, one) = constants(newVar, 0.0, 1.0, type = Types.bool, put = put)

        switch(
            listOf(),
            newVar,
            lt,
            listOf(),
            one to nop, // index < dim
            zero to panic, // index >= dim
            put = put,
        )
    }
}