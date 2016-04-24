abstract JointType

immutable Joint
    name::ASCIIString
    frameBefore::CartesianFrame3D
    frameAfter::CartesianFrame3D
    jointType::JointType

    Joint(name::ASCIIString, jointType::JointType) = new(name, CartesianFrame3D(string("before_", name)), CartesianFrame3D(string("after_", name)), jointType)
end
show(io::IO, joint::Joint) = print(io, "Joint \"$(joint.name)\": $(joint.jointType)")
showcompact(io::IO, joint::Joint) = print(io, "$(joint.name)")

immutable QuaternionFloating <: JointType
    motionSubspace::Array{Float64, 2}
    function QuaternionFloating()
        new(eye(6))
    end
end
show(io::IO, jt::QuaternionFloating) = print(io, "Quaternion floating joint")

function joint_transform{T<:Real}(j::Joint, q::Vector{T}, jt::QuaternionFloating = j.jointType)
    rot = Quaternion(q[1], q[2 : 4])
    trans = Vec(q[5 : 7])
    return Transform3D{T}(j.frameAfter, j.frameBefore, rot, trans)
end

function motion_subspace{T<:Real}(j::Joint, q::Vector{T}, jt::QuaternionFloating = j.jointType)
    return GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, copy(jt.motionSubspace))
end

num_positions(j::Joint, jt::QuaternionFloating = j.jointType) = 7::Int64
num_velocities(j::Joint, jt::QuaternionFloating = j.jointType) = 6::Int64
has_fixed_motion_subspace(j::Joint, jt::QuaternionFloating = j.jointType) = true
bias_acceleration{T<:Real}(j::Joint, q::Vector{T}, v::Vector{T}, jt::QuaternionFloating = j.jointType) = zero(SpatialAcceleration{T}, j.frameAfter, j.frameBefore, j.frameAfter)

function zero_configuration{T<:Real}(j::Joint, ::Type{T}, jt::QuaternionFloating = j.jointType)
    return [one(T); zeros(T, 6)]
end
function rand_configuration{T<:Real}(j::Joint, ::Type{T}, jt::QuaternionFloating = j.jointType)
    quat = nquatrand() # TODO: only works when T == Float64
    return [quat.s; quat.v1; quat.v2; quat.v3; rand(T, 3)]
end

function joint_twist{T<:Real}(j::Joint, q::Vector{T}, v::Vector{T}, jt::QuaternionFloating = j.jointType)
    return Twist(j.frameAfter, j.frameBefore, j.frameAfter, Vec(v[1 : 3]), Vec(v[4 : 6]))
end

abstract OneDegreeOfFreedomFixedAxis <: JointType

immutable Prismatic{T<:Real} <: OneDegreeOfFreedomFixedAxis
    translation_axis::Vec{3, T}
    motionSubspace::Vector{Float64}
    Prismatic(translation_axis::Vec{3, T}) = new(translation_axis, [zeros(3); Array(translation_axis)])
end
Prismatic{T}(rotation_axis::Vec{3, T}) = Prismatic{T}(rotation_axis)
show(io::IO, jt::Prismatic) = print(io, "Prismatic joint with axis $(jt.translation_axis)")

joint_transform{T1<:Real, T2}(j::Joint, q::Vector{T1}, jt::Prismatic{T2} = j.jointType) = Transform3D(j.frameAfter, j.frameBefore, q[1] * jt.translation_axis)

function joint_twist{T<:Real}(j::Joint, q::Vector{T}, v::Vector{T}, jt::Prismatic = j.jointType)
    return Twist(j.frameAfter, j.frameBefore, j.frameAfter, zero(Vec{3, T}), jt.translation_axis * v[1])
end

function motion_subspace{T<:Real}(j::Joint, q::Vector{T}, jt::Prismatic = j.jointType)
    return GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, copy(jt.motionSubspace))
end

immutable Revolute{T<:Real} <: OneDegreeOfFreedomFixedAxis
    rotation_axis::Vec{3, T}
    motionSubspace::Vector{Float64}
    Revolute(rotation_axis::Vec{3, T}) = new(rotation_axis, [Array(rotation_axis); zeros(3)])
end
Revolute{T}(rotation_axis::Vec{3, T}) = Revolute{T}(rotation_axis)
show(io::IO, jt::Revolute) = print(io, "Revolute joint with axis $(jt.rotation_axis)")

joint_transform{T1, T2}(j::Joint, q::Vector{T1}, jt::Revolute{T2} = j.jointType) = Transform3D(j.frameAfter, j.frameBefore, qrotation(Array(jt.rotation_axis), q[1]))

function joint_twist{T<:Real}(j::Joint, q::Vector{T}, v::Vector{T}, jt::Revolute = j.jointType)
    return Twist(j.frameAfter, j.frameBefore, j.frameAfter, jt.rotation_axis * v[1], zero(Vec{3, T}))
end

function motion_subspace{T<:Real}(j::Joint, q::Vector{T}, jt::Revolute = j.jointType)
    return GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, copy(jt.motionSubspace))
end

num_positions(j::Joint, jt::OneDegreeOfFreedomFixedAxis = j.jointType) = 1::Int64
num_velocities(j::Joint, jt::OneDegreeOfFreedomFixedAxis = j.jointType) = 1::Int64
zero_configuration{T<:Real}(j::Joint, ::Type{T}, jt::OneDegreeOfFreedomFixedAxis = j.jointType) = [zero(T)]
rand_configuration{T<:Real}(j::Joint, ::Type{T}, jt::OneDegreeOfFreedomFixedAxis = j.jointType) = [rand(T)]
has_fixed_motion_subspace(j::Joint, jt::OneDegreeOfFreedomFixedAxis = j.jointType) = true
bias_acceleration{T<:Real}(j::Joint, q::Vector{T}, v::Vector{T}, jt::OneDegreeOfFreedomFixedAxis = j.jointType) = zero(SpatialAcceleration{T}, j.frameAfter, j.frameBefore, j.frameAfter)

num_positions(itr) = reduce((val, joint) -> val + num_positions(joint), 0, itr)
num_velocities(itr) = reduce((val, joint) -> val + num_velocities(joint), 0, itr)
