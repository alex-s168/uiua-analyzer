package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.ir.IrBlock

fun function(
    name: String,
    private: Boolean,
    args: List<Pair<MLIRVar, MLIRType>>,
    rets: List<Pair<MLIRVar, MLIRType>>,
    body: List<String>
): String {
    val res = StringBuilder()

    res.append("func.func")
    if (!private)
        res.append(" private") // means export for some reason
    res.append(" @")
    res.append(name.legalizeMLIR())
    res.append('(')
    args.forEachIndexed { index, (va, ty) ->
        if (index > 0)
            res.append(", ")
        res.append(va)
        res.append(": ")
        res.append(ty)
    }
    res.append(") -> (")
    rets.forEachIndexed { index, (_, ty) ->
        if (index > 0)
            res.append(", ")
        res.append(ty)
    }
    res.append(") {\n")

    body.flatMap { it.split('\n') }.forEach {
        res.append("  ")
        res.append(it)
        res.append('\n')
    }

    res.append("  return ")
    rets.forEachIndexed { index, (va, _) ->
        if (index > 0)
            res.append(", ")
        res.append(va)
    }
    if (rets.isNotEmpty())
        res.append(" : ")
    rets.forEachIndexed { index, (_, ty) ->
        if (index > 0)
            res.append(", ")
        res.append(ty)
    }
    res.append("\n}")

    return res.toString()
}