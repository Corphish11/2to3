from numpy import linalg as LA


def parse_tracklet_heads(tracklet_heads, drop_short_tracklet=True, short_thresh=5):
    if drop_short_tracklet:
        long_trajectories = []
        for trajectory_head in tracklet_heads:
            l = 0
            cur = trajectory_head
            while cur:
                cur = cur.next
                l += 1

            if l > short_thresh:
                long_trajectories.append(trajectory_head)
        tracklet_heads = long_trajectories

    trajectories = []
    for i, trajectory_head in enumerate(tracklet_heads):
        trajectory = {}
        while trajectory_head:
            trajectory[trajectory_head.frame_index] = trajectory_head
            trajectory_head.track_id = i
            trajectory_head = trajectory_head.next

        trajectories.append(trajectory)
    return trajectories


def correspondence_cost(rect1, rect2):
    dis = LA.norm(rect1.center - rect2.center)
    area_ratio = max(rect1.area / (rect2.area + 1e-5), rect2.area / (rect1.area + 1e-5))
    return dis * area_ratio


def track_balls(frame_based_results, max_frame_difference = 3, radius_parameter=1.0):
    for i in range(1, max_frame_difference+1):
        correspondences = []
        for t, (pre_frame_rect, next_frame_rect) in enumerate(zip(frame_based_results[:-i],
                                                                    frame_based_results[i:])):
            for j, previous_rect in enumerate(pre_frame_rect):
                for k, next_rect in enumerate(next_frame_rect):
                    if not previous_rect.next and not next_rect.previous:
                        correspondences.append(
                            (
                                (previous_rect, next_rect), correspondence_cost(previous_rect, next_rect)
                            )
                        )

        sorted_correspondences = sorted(correspondences, key=lambda x: x[1])

        for c in sorted_correspondences:
            (previous_rect, next_rect), cost = c

            if cost < (previous_rect.radius + next_rect.radius) / 2 * i * radius_parameter:
                previous_rect.next = next_rect
                next_rect.previous = previous_rect

    tracklet_heads = []
    for i, frame_rects in enumerate(frame_based_results):
        for j, rect in enumerate(frame_rects):
            if not rect.previous:
                tracklet_heads.append(rect)

    return tracklet_heads
