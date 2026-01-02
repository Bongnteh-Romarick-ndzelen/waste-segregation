"""
robotic_arm.py
Robotic arm visualization for waste segregation system
"""

import pygame
import math
import numpy as np
import time
import threading
import queue
from enum import Enum

class ArmState(Enum):
    """Robotic arm states"""
    IDLE = 0
    MOVING_TO_ITEM = 1
    PICKING_ITEM = 2
    MOVING_TO_BIN = 3
    DROPPING_ITEM = 4
    RETURNING_HOME = 5

class RoboticArm:
    def __init__(self, width=800, height=600):
        """Initialize robotic arm visualizer"""
        pygame.init()
        
        # Display settings
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robotic Arm - Waste Sorting System")
        
        # Colors
        self.colors = {
            'background': (40, 40, 40),
            'conveyor': (60, 60, 60),
            'arm_base': (100, 100, 100),
            'arm_segment': (80, 120, 200),
            'arm_segment_highlight': (100, 140, 220),
            'joint': (200, 100, 50),
            'gripper': (220, 60, 60),
            'text': (255, 255, 255),
            'text_highlight': (255, 255, 0),
            'bin_outline': (200, 200, 200),
            'status_idle': (100, 100, 100),
            'status_moving': (255, 255, 0),
            'status_picking': (0, 255, 0),
            'status_dropping': (255, 0, 0)
        }
        
        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Arm parameters
        self.base_pos = (width // 2, height - 100)
        self.arm_lengths = [120, 100, 80, 60]  # Four segments
        self.arm_widths = [20, 16, 12, 10]     # Width of each segment
        self.joint_radii = [15, 12, 10, 8, 6]  # Radii for joints (including end effector)
        
        # Joint angles (in degrees)
        self.angles = [45, -30, -20, -10]  # Initial angles
        self.target_angles = self.angles.copy()
        
        # Arm state
        self.state = ArmState.IDLE
        self.state_progress = 0.0  # 0 to 1 for animation progress
        self.animation_speed = 0.02
        
        # Waste handling
        self.current_item = None
        self.item_class = None
        self.item_position = None
        self.item_in_gripper = False
        
        # Bins for different waste types
        self.bins = self.create_bins()
        self.bin_contents = {bin_name: 0 for bin_name in self.bins.keys()}
        
        # Control
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Command queue
        self.command_queue = queue.Queue()
        
        # Statistics
        self.statistics = {
            'total_sorted': 0,
            'sorting_time': 0,
            'start_time': time.time(),
            'last_sort_time': None
        }
        
        # Initialize inverse kinematics solver
        self.ik_solver = InverseKinematicsSolver(self.arm_lengths)
        
        print("Robotic Arm Visualizer initialized")
    
    def create_bins(self):
        """Create waste bins for different materials"""
        bin_width = 80
        bin_height = 60
        bin_spacing = 10
        start_x = 50
        
        bins = {
            'Cardboard': {
                'pos': (start_x, self.height - 80),
                'color': (139, 69, 19),
                'target_angles': [60, -40, -15, -5]  # Pre-calculated angles for bin
            },
            'Glass': {
                'pos': (start_x + bin_width + bin_spacing, self.height - 80),
                'color': (0, 180, 0),
                'target_angles': [30, -50, -10, -10]
            },
            'Metal': {
                'pos': (start_x + 2*(bin_width + bin_spacing), self.height - 80),
                'color': (128, 128, 128),
                'target_angles': [0, -60, -5, -15]
            },
            'Paper': {
                'pos': (start_x + 3*(bin_width + bin_spacing), self.height - 80),
                'color': (220, 220, 220),
                'target_angles': [-30, -50, 0, -20]
            },
            'Plastic': {
                'pos': (start_x + 4*(bin_width + bin_spacing), self.height - 80),
                'color': (200, 0, 0),
                'target_angles': [-60, -40, 5, -25]
            },
            'Trash': {
                'pos': (self.width - 150, self.height - 80),
                'color': (128, 0, 128),
                'target_angles': [90, -20, -20, 0]
            }
        }
        
        # Conveyor position (where items come from)
        self.conveyor_pos = (self.width // 2, self.height // 2)
        
        return bins
    
    def calculate_forward_kinematics(self):
        """Calculate arm endpoint position from angles"""
        x, y = self.base_pos
        current_angle = 0
        joint_positions = [(x, y)]
        
        for i, (length, angle_deg) in enumerate(zip(self.arm_lengths, self.angles)):
            current_angle += math.radians(angle_deg)
            x += length * math.cos(current_angle)
            y += length * math.sin(current_angle)
            joint_positions.append((x, y))
        
        return joint_positions
    
    def move_to_target_smooth(self, target_angles):
        """Smoothly move arm to target angles"""
        # Simple linear interpolation for smooth movement
        for i in range(len(self.angles)):
            angle_diff = target_angles[i] - self.angles[i]
            
            # Normalize angle difference to shortest path
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            
            # Move with speed limit
            max_speed = 2.0  # degrees per frame
            move_amount = max(-max_speed, min(max_speed, angle_diff))
            self.angles[i] += move_amount
            
            # Normalize angle to [-180, 180]
            if self.angles[i] > 180:
                self.angles[i] -= 360
            elif self.angles[i] < -180:
                self.angles[i] += 360
        
        # Check if we've reached target (within tolerance)
        tolerance = 0.5  # degrees
        reached = all(abs(self.angles[i] - target_angles[i]) < tolerance 
                     for i in range(len(self.angles)))
        
        return reached
    
    def pick_item(self, item_class, item_position=None):
        """Initiate picking sequence for a waste item"""
        if self.state != ArmState.IDLE:
            print(f"Cannot pick item, arm is busy: {self.state}")
            return False
        
        self.state = ArmState.MOVING_TO_ITEM
        self.state_progress = 0.0
        self.item_class = item_class
        self.item_position = item_position if item_position else self.conveyor_pos
        self.item_in_gripper = False
        
        # Calculate target angles for item position using IK
        target_pos = self.item_position
        success, target_angles = self.ik_solver.solve(target_pos, self.base_pos)
        
        if success:
            self.target_angles = target_angles
        else:
            # Fallback to pre-defined angles for conveyor
            self.target_angles = [0, -45, -30, -15]
        
        print(f"Starting to pick {list(self.bins.keys())[item_class]} item")
        return True
    
    def update(self):
        """Update arm state and animation"""
        # Process commands from queue
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                if command[0] == 'pick':
                    _, item_class, position = command
                    self.pick_item(item_class, position)
                elif command[0] == 'reset':
                    self.reset_arm()
            except queue.Empty:
                break
        
        # State machine for arm operation
        if self.state == ArmState.IDLE:
            # Idle state - slight breathing animation
            time_passed = time.time() - self.statistics['start_time']
            breath_amount = math.sin(time_passed * 0.5) * 0.5
            self.angles[3] = -10 + breath_amount * 5  # Wrist slight movement
        
        elif self.state == ArmState.MOVING_TO_ITEM:
            # Move to item position
            reached = self.move_to_target_smooth(self.target_angles)
            if reached:
                self.state = ArmState.PICKING_ITEM
                self.state_progress = 0.0
                print("Reached item position, starting pick")
        
        elif self.state == ArmState.PICKING_ITEM:
            # Picking animation
            self.state_progress += self.animation_speed
            if self.state_progress >= 1.0:
                self.state_progress = 1.0
                self.item_in_gripper = True
                self.state = ArmState.MOVING_TO_BIN
                self.state_progress = 0.0
                
                # Calculate target angles for appropriate bin
                bin_name = list(self.bins.keys())[self.item_class]
                self.target_angles = self.bins[bin_name]['target_angles']
                print(f"Item picked, moving to {bin_name} bin")
        
        elif self.state == ArmState.MOVING_TO_BIN:
            # Move to bin
            reached = self.move_to_target_smooth(self.target_angles)
            if reached:
                self.state = ArmState.DROPPING_ITEM
                self.state_progress = 0.0
                print("Reached bin, dropping item")
        
        elif self.state == ArmState.DROPPING_ITEM:
            # Dropping animation
            self.state_progress += self.animation_speed * 1.5  # Faster dropping
            if self.state_progress >= 1.0:
                self.state_progress = 1.0
                self.item_in_gripper = False
                
                # Update bin contents
                bin_name = list(self.bins.keys())[self.item_class]
                self.bin_contents[bin_name] += 1
                
                # Update statistics
                self.statistics['total_sorted'] += 1
                self.statistics['last_sort_time'] = time.time()
                
                self.state = ArmState.RETURNING_HOME
                self.state_progress = 0.0
                
                # Target angles for home position
                self.target_angles = [45, -30, -20, -10]
                print(f"Item dropped in {bin_name} bin")
        
        elif self.state == ArmState.RETURNING_HOME:
            # Return to home position
            reached = self.move_to_target_smooth(self.target_angles)
            if reached:
                self.state = ArmState.IDLE
                self.item_class = None
                self.item_position = None
                print("Returned to home position")
    
    def draw_conveyor(self):
        """Draw conveyor belt representation"""
        conveyor_width = 400
        conveyor_height = 40
        conveyor_x = self.width // 2 - conveyor_width // 2
        conveyor_y = self.height // 2 - conveyor_height // 2
        
        # Conveyor base
        pygame.draw.rect(self.screen, self.colors['conveyor'],
                        (conveyor_x, conveyor_y, conveyor_width, conveyor_height))
        
        # Conveyor lines (moving animation)
        line_spacing = 40
        line_offset = int((time.time() * 20) % line_spacing)
        
        for x in range(conveyor_x - line_offset, conveyor_x + conveyor_width, line_spacing):
            pygame.draw.line(self.screen, (100, 100, 100),
                           (x, conveyor_y),
                           (x, conveyor_y + conveyor_height), 2)
        
        # Conveyor border
        pygame.draw.rect(self.screen, (150, 150, 150),
                        (conveyor_x, conveyor_y, conveyor_width, conveyor_height), 2)
        
        # Conveyor label
        label = self.font_small.render("CONVEYOR BELT", True, (200, 200, 200))
        self.screen.blit(label, (conveyor_x + conveyor_width // 2 - 50, conveyor_y - 25))
    
    def draw_bins(self):
        """Draw waste bins"""
        for bin_name, bin_data in self.bins.items():
            x, y = bin_data['pos']
            color = bin_data['color']
            count = self.bin_contents[bin_name]
            
            # Bin body
            bin_width, bin_height = 80, 60
            pygame.draw.rect(self.screen, color,
                           (x - bin_width//2, y - bin_height, bin_width, bin_height))
            
            # Bin outline
            pygame.draw.rect(self.screen, self.colors['bin_outline'],
                           (x - bin_width//2, y - bin_height, bin_width, bin_height), 2)
            
            # Bin label
            label = self.font_small.render(bin_name, True, (255, 255, 255))
            label_rect = label.get_rect(center=(x, y - bin_height//2))
            self.screen.blit(label, label_rect)
            
            # Item count
            count_text = self.font_small.render(f"Count: {count}", True, (255, 255, 255))
            count_rect = count_text.get_rect(center=(x, y - bin_height//2 + 20))
            self.screen.blit(count_text, count_rect)
    
    def draw_arm(self):
        """Draw the robotic arm"""
        # Get joint positions from forward kinematics
        joint_positions = self.calculate_forward_kinematics()
        
        # Draw arm base
        base_radius = 25
        pygame.draw.circle(self.screen, self.colors['arm_base'],
                          self.base_pos, base_radius)
        pygame.draw.circle(self.screen, (50, 50, 50),
                          self.base_pos, base_radius, 3)
        
        # Draw arm segments
        for i in range(len(self.arm_lengths)):
            start_pos = joint_positions[i]
            end_pos = joint_positions[i + 1]
            
            # Arm segment with gradient
            segment_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            segment_length = math.sqrt(segment_vector[0]**2 + segment_vector[1]**2)
            
            if segment_length > 0:
                # Draw main segment
                pygame.draw.line(self.screen, self.colors['arm_segment'],
                               start_pos, end_pos, self.arm_widths[i])
                
                # Draw highlight on top
                highlight_width = max(2, self.arm_widths[i] // 3)
                pygame.draw.line(self.screen, self.colors['arm_segment_highlight'],
                               start_pos, end_pos, highlight_width)
        
        # Draw joints
        for i, pos in enumerate(joint_positions):
            radius = self.joint_radii[i]
            
            # Joint body
            pygame.draw.circle(self.screen, self.colors['joint'],
                              (int(pos[0]), int(pos[1])), radius)
            
            # Joint outline
            pygame.draw.circle(self.screen, (50, 50, 50),
                              (int(pos[0]), int(pos[1])), radius, 2)
            
            # Joint highlight
            highlight_radius = max(2, radius // 3)
            pygame.draw.circle(self.screen, (255, 200, 100),
                              (int(pos[0]), int(pos[1])), highlight_radius)
        
        # Draw gripper (end effector)
        end_pos = joint_positions[-1]
        gripper_radius = 10
        
        # Gripper state based on animation
        if self.state == ArmState.PICKING_ITEM:
            # Closing gripper animation
            gripper_openness = 1.0 - self.state_progress
        elif self.state == ArmState.DROPPING_ITEM:
            # Opening gripper animation
            gripper_openness = self.state_progress
        else:
            # Idle or carrying
            gripper_openness = 0.3 if self.item_in_gripper else 0.7
        
        # Draw gripper
        gripper_color = self.colors['gripper']
        if self.item_in_gripper:
            gripper_color = (255, 100, 100)  # Brighter when holding item
        
        pygame.draw.circle(self.screen, gripper_color,
                          (int(end_pos[0]), int(end_pos[1])), gripper_radius)
        
        # Draw gripper jaws
        jaw_length = 15
        jaw_angle = 30 * gripper_openness  # 0 to 30 degrees
        
        # Left jaw
        left_angle = math.radians(180 + jaw_angle)
        left_jaw_end = (
            end_pos[0] + jaw_length * math.cos(left_angle),
            end_pos[1] + jaw_length * math.sin(left_angle)
        )
        pygame.draw.line(self.screen, (50, 50, 50),
                        end_pos, left_jaw_end, 3)
        
        # Right jaw
        right_angle = math.radians(-jaw_angle)
        right_jaw_end = (
            end_pos[0] + jaw_length * math.cos(right_angle),
            end_pos[1] + jaw_length * math.sin(right_angle)
        )
        pygame.draw.line(self.screen, (50, 50, 50),
                        end_pos, right_jaw_end, 3)
        
        # Draw item in gripper if carrying
        if self.item_in_gripper and self.item_class is not None:
            item_radius = 8
            item_color = self.bins[list(self.bins.keys())[self.item_class]]['color']
            
            # Draw item
            pygame.draw.circle(self.screen, item_color,
                             (int(end_pos[0]), int(end_pos[1])), item_radius)
            pygame.draw.circle(self.screen, (255, 255, 255),
                             (int(end_pos[0]), int(end_pos[1])), item_radius, 1)
            
            # Draw item label
            item_name = list(self.bins.keys())[self.item_class]
            item_label = self.font_small.render(item_name, True, (255, 255, 255))
            self.screen.blit(item_label, (end_pos[0] + 15, end_pos[1] - 10))
    
    def draw_status_panel(self):
        """Draw status panel with information"""
        panel_width = 300
        panel_height = 150
        panel_x = self.width - panel_width - 20
        panel_y = 20
        
        # Panel background
        pygame.draw.rect(self.screen, (0, 0, 0, 180),
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, (100, 100, 100),
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Status title
        title = self.font_medium.render("ROBOTIC ARM STATUS", True, self.colors['text_highlight'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Current state
        state_colors = {
            ArmState.IDLE: self.colors['status_idle'],
            ArmState.MOVING_TO_ITEM: self.colors['status_moving'],
            ArmState.PICKING_ITEM: self.colors['status_picking'],
            ArmState.MOVING_TO_BIN: self.colors['status_moving'],
            ArmState.DROPPING_ITEM: self.colors['status_dropping'],
            ArmState.RETURNING_HOME: self.colors['status_moving']
        }
        
        state_text = self.state.name.replace('_', ' ').title()
        state_color = state_colors.get(self.state, self.colors['text'])
        
        state_label = self.font_small.render(f"State: {state_text}", True, state_color)
        self.screen.blit(state_label, (panel_x + 10, panel_y + 40))
        
        # Current item
        if self.item_class is not None:
            item_name = list(self.bins.keys())[self.item_class]
            item_text = f"Current Item: {item_name}"
        else:
            item_text = "Current Item: None"
        
        item_label = self.font_small.render(item_text, True, self.colors['text'])
        self.screen.blit(item_label, (panel_x + 10, panel_y + 65))
        
        # Statistics
        stats_y = panel_y + 90
        stats = [
            f"Total Sorted: {self.statistics['total_sorted']}",
            f"FPS: {int(self.clock.get_fps())}"
        ]
        
        for i, stat in enumerate(stats):
            stat_label = self.font_small.render(stat, True, self.colors['text'])
            self.screen.blit(stat_label, (panel_x + 10, stats_y + i * 20))
    
    def draw_instructions(self):
        """Draw control instructions"""
        instructions = [
            "CONTROLS:",
            "0-5: Pick waste item (0=Cardboard, 1=Glass, etc.)",
            "R: Reset arm position",
            "SPACE: Test random item",
            "Q: Quit"
        ]
        
        y_pos = self.height - 120
        for i, instruction in enumerate(instructions):
            color = self.colors['text_highlight'] if i == 0 else self.colors['text']
            instruction_label = self.font_small.render(instruction, True, color)
            self.screen.blit(instruction_label, (20, y_pos + i * 20))
    
    def draw(self):
        """Draw all components"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw components
        self.draw_conveyor()
        self.draw_bins()
        self.draw_arm()
        self.draw_status_panel()
        self.draw_instructions()
        
        # Update display
        pygame.display.flip()
    
    def reset_arm(self):
        """Reset arm to home position"""
        self.state = ArmState.RETURNING_HOME
        self.state_progress = 0.0
        self.target_angles = [45, -30, -20, -10]
        self.item_in_gripper = False
        print("Arm reset to home position")
    
    def run_visualization(self):
        """Main visualization loop"""
        print("Starting robotic arm visualization...")
        print("Controls: 0-5 = Pick waste type, R = Reset, SPACE = Random, Q = Quit")
        
        while self.is_running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Quit
                        self.is_running = False
                    
                    elif event.key == pygame.K_r:  # Reset
                        self.reset_arm()
                    
                    elif event.key == pygame.K_SPACE:  # Test random item
                        random_class = np.random.randint(0, 6)
                        self.pick_item(random_class)
                    
                    elif pygame.K_0 <= event.key <= pygame.K_5:  # Pick specific waste
                        waste_class = event.key - pygame.K_0
                        self.pick_item(waste_class)
            
            # Update arm state
            self.update()
            
            # Draw everything
            self.draw()
            
            # Control frame rate
            self.clock.tick(self.fps)
        
        pygame.quit()
        print("Robotic arm visualization stopped")

class InverseKinematicsSolver:
    """Inverse kinematics solver for robotic arm"""
    
    def __init__(self, arm_lengths):
        self.arm_lengths = arm_lengths
        self.num_joints = len(arm_lengths)
    
    def solve(self, target_pos, base_pos, max_iterations=100, tolerance=1.0):
        """Solve inverse kinematics using CCD (Cyclic Coordinate Descent)"""
        # Convert to numpy arrays
        target = np.array(target_pos, dtype=float)
        base = np.array(base_pos, dtype=float)
        
        # Initialize angles (all zeros)
        angles = np.zeros(self.num_joints)
        
        # CCD algorithm
        for iteration in range(max_iterations):
            # Forward kinematics from end to base
            current_pos = base.copy()
            current_angle = 0
            
            for i in range(self.num_joints):
                current_angle += np.deg2rad(angles[i])
                current_pos[0] += self.arm_lengths[i] * np.cos(current_angle)
                current_pos[1] += self.arm_lengths[i] * np.sin(current_angle)
            
            # Check if we're close enough
            error = np.linalg.norm(target - current_pos)
            if error < tolerance:
                return True, angles.tolist()
            
            # Backward iteration from end effector to base
            for i in reversed(range(self.num_joints)):
                # Calculate current end effector position from this joint onward
                joint_pos = base.copy()
                joint_angle = 0
                
                for j in range(i):
                    joint_angle += np.deg2rad(angles[j])
                    joint_pos[0] += self.arm_lengths[j] * np.cos(joint_angle)
                    joint_pos[1] += self.arm_lengths[j] * np.sin(joint_angle)
                
                # Calculate vectors
                current_effector_pos = current_pos - joint_pos
                target_effector_pos = target - joint_pos
                
                # Calculate angle between vectors
                current_angle = np.arctan2(current_effector_pos[1], current_effector_pos[0])
                target_angle = np.arctan2(target_effector_pos[1], target_effector_pos[0])
                
                # Update joint angle
                angle_diff = target_angle - current_angle
                angles[i] += np.rad2deg(angle_diff)
                
                # Normalize angle
                angles[i] = (angles[i] + 180) % 360 - 180
                
                # Clamp angle to reasonable limits
                angles[i] = max(-150, min(150, angles[i]))
                
                # Recalculate forward kinematics
                current_pos = base.copy()
                current_angle = 0
                
                for j in range(self.num_joints):
                    current_angle += np.deg2rad(angles[j])
                    current_pos[0] += self.arm_lengths[j] * np.cos(current_angle)
                    current_pos[1] += self.arm_lengths[j] * np.sin(current_angle)
        
        # Return best solution found
        return False, angles.tolist()

def test_robotic_arm():
    """Test the robotic arm visualization"""
    print("Testing Robotic Arm Visualization...")
    
    # Create arm
    arm = RoboticArm(width=1000, height=700)
    
    # Run visualization
    arm.run_visualization()
    
    print("Robotic arm test completed!")

def main():
    """Main function for standalone testing"""
    test_robotic_arm()

if __name__ == "__main__":
    main()