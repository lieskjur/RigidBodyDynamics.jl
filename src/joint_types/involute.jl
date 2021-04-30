struct Involute{T}
	i::SVector{3, T}
	j::SVector{3, T}
	k::SVector{3, T}
	ρ::T
	ρ_inv::T

	function Involute{T}(b_axis::AbstractVector,n_axis::AbstractVector,ρ::T) where {T}
		j, k = map(axis -> normalize(SVector{3}(axis)), (n_axis, b_axis))
        @assert isapprox(k ⋅ j, 0; atol = 100 * eps(T))
        @assert ρ > 0
        new{T}(j × k, j, k, ρ, inv(ρ))
    end
end

@propagate_inbounds function joint_transform(jt::Involute, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D, q::AbstractVector)
	rot = RotMatrix(AngleAxis(jt.ρ_inv*q[1],jt.k[1],jt.k[2],jt.k[3],false))
	trans = rot*(jt.ρ*jt.j+q[1]*jt.i)
	Transform3D(frame_after,frame_before,rot,trans)
end

@propagate_inbounds function joint_twist(jt::Involute, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector, v::AbstractVector)
	angular = jt.ρ_inv*jt.k*v[1]
	linear = jt.ρ_inv*q[1]*jt.j*v[1]
	Twist(frame_after, frame_before, frame_after, angular, linear)
end

@propagate_inbounds function joint_spacial_acceleration(jt::Involute, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector, v::AbstractVector, vd::AbstractVector),q[1],v[1],vd[1])
	S = promote_eltype(jt, q, v, vd)
	angular = jt.ρ_inv*jt.k*vd[1]
	linear = (jt.ρ_inv*q[1]*jt.j)*vd[1] + (jt.j-jt.ρ_inv*q[1]*jt.i)*jt.ρ_inv*v[1]^2
	SpatialAcceleration{S}(frame_after, frame_before, frame_after, angular, linear)
end

@propagate_inbounds function bias_acceleration(jt::Planar, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector, v::AbstractVector)
    S = promote_eltype(jt, q, v)
    linear = zero(SVector{3,S})
    angular = (jt.j-jt.ρ_inv*q[1]*jt.i)*jt.ρ_inv*v[1]^2
    SpatialAcceleration{S}(frame_after, frame_before, frame_after, angular,linear)
end

@propagate_inbounds function joint_torque!(τ::AbstractVector, jt::Planar, q::AbstractVector, joint_wrench::Wrench)
	R = RotMatrix(AngleAxis(-jt.ρ_inv*q[1],jt.k[1],jt.k[2],jt.k[3],false))
	τ[1] = jt.ρ_inv*q[1]*jt.j'*linear(joint_wrench) + ρ_inv*k'*angular(joint_wrench)
	nothing
end

@inline function motion_subspace(jt::Planar, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector)
    S = promote_eltype(jt, q)
	angular = hcat(zero(SMatrix{3,2,S}),jt.k)
	linear = hcat(zero(SVector{3,S}),jt.j,zero(SVector{3,S}))
	GeometricJacobian(frame_after, frame_before, frame_after, angular, linear)
end

@inline function constraint_wrench_subspace(jt::Planar, joint_transform::Transform3D)
    S = promote_eltype(jt, joint_transform)
    angular = hcat(jt.i, jt.j, zero(SVector{3, S}))
    linear = hcat(jt.i, zero(SVector{3, S}), jt.k )
    WrenchMatrix(joint_transform.from, angular, linear)
end