package me.alex_s168.uiua.ir

import me.alex_s168.uiua.Type

data class IrVar(
    val type: Type,
    val id: ULong
) {
    override fun hashCode() = id.hashCode()

    override fun equals(other: Any?): Boolean =
        other is IrVar && other.id == id

    override fun toString() =
        "$type %$id"
}